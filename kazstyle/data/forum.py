"""Collect bounded public question bodies for review, keeping thread provenance."""
import argparse
import json
import html
import re
import time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urljoin,urlsplit,unquote
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup
from kazstyle.data.corpus import digest,file_hash,write_json,write_jsonl,duplicate_groups
from kazstyle.data.quality import clean_text,inspect_text

BASE='https://surak.baribar.kz/'
CATEGORIES=['компьютер-туралы','ұялы телефон','музыка-ән','кино-теледидар-телешоу']


def own_post_element(post, selector):
    """Find a post's own element without borrowing from nested replies."""
    post_classes = {'qa-q-view', 'qa-a-list-item', 'qa-c-list-item'}
    for element in post.select(selector):
        owner = next((parent for parent in element.parents
                      if post_classes.intersection(parent.get('class', []))), None)
        if owner is post:
            return element
    return None


def post_timestamps(post):
    created = own_post_element(post, 'time[itemprop="dateCreated"][datetime]')
    modified = own_post_element(post, 'time[itemprop="dateModified"][datetime]')
    return (created.get('datetime') if created else None,
            modified.get('datetime') if modified else None)


def has_contact_identifier(text):
    """Screen local/international phone-like numbers and possible 12-digit IDs.

    Digit boundaries deliberately also match numbers attached to words. This is
    a conservative review filter, not a claim that every match is personal data.
    """
    return bool(re.search(
        r'(?<!\d)(?:\+?7|8)(?:[\s()-]*\d){10}(?!\d)|'
        r'(?<!\w)\+[1-9](?:[\s().-]*\d){6,14}(?![\s().-]*\d)|'
        r'(?<!\d)\d{12}(?!\d)', text))


def parse_question(content):
    soup=BeautifulSoup(content,'html.parser')
    body=soup.select_one('.qa-q-view-content [itemprop="text"]')
    title=soup.find('h1')
    if body is None or title is None:raise ValueError('Question body/title missing')
    for node in body.select('script,style,nav,form,button'):node.decompose()
    question=body.get_text(' ',strip=True);heading=title.get_text(' ',strip=True)
    # Escaped Word fragment comments can survive HTML text extraction.
    question=re.sub(r'<!--.*?-->',' ',html.unescape(question),flags=re.S)
    heading=re.sub(r'<!--.*?-->',' ',html.unescape(heading),flags=re.S)
    normalize=lambda text:' '.join(re.findall(r'\w+',text.casefold()))
    if normalize(heading) not in normalize(question):question=heading+'\n'+question
    author=soup.select_one('.qa-q-view-who-data .qa-user-link[href]')
    author_group=digest('surak_author:'+urljoin(BASE,author['href']))[:24] if author else None
    return question,author_group


def rescreen(source,out):
    """Re-extract saved pages, then quarantine near duplicates; no new requests."""
    import pandas as pd
    if out.exists():raise FileExistsError(out)
    original=[]
    for name in ['candidates.jsonl','quarantine.jsonl']:
        original.extend(json.loads(line) for line in (source/name).read_text(encoding='utf-8').splitlines())
    accepted,rejected,changes=[],[],[]
    for row in original:
        snapshot=Path(row['source_snapshot'])
        if file_hash(snapshot)!=row['source_sha256']:raise ValueError('Source snapshot changed')
        raw,author=parse_question(snapshot.read_bytes())
        text,cleaning=clean_text(raw);stats,reasons=inspect_text(text,min_words=20,max_words=500)
        if has_contact_identifier(text):reasons.append('possible_contact_or_identifier')
        record={**row,'text':text,'author_group':author,'quality':stats,'cleaning':cleaning,
                'content_hash':digest(text.casefold()),'eligibility_reasons':reasons}
        if text!=row['text']:changes.append({'doc_id':row['doc_id'],'before_words':len(row['text'].split()),'after_words':len(text.split())})
        (rejected if reasons else accepted).append(record)
    frame=pd.DataFrame(accepted);frame['excerpt_hash']=frame.content_hash;frame['duplicate_probe']=frame.text
    groups,edges=duplicate_groups(frame,.92)
    unique=[];seen={}
    for row,group in zip(accepted,groups):
        row['duplicate_group_id']=group
        if group in seen:
            rejected.append({**row,'eligibility_reasons':['near_duplicate'],'duplicate_of':seen[group]})
        else:seen[group]=row['doc_id'];unique.append(row)
    out.mkdir(parents=True)
    write_jsonl(out/'candidates.jsonl',unique);write_jsonl(out/'quarantine.jsonl',rejected)
    summary={'source':str(source),'source_hashes':{name:file_hash(source/name) for name in ['candidates.jsonl','quarantine.jsonl']},
             'candidates':len(unique),'quarantined':len(rejected),'changed_extractions':changes,'near_duplicate_edges':len(edges),
             'reasons':dict(Counter(reason for r in rejected for reason in r['eligibility_reasons'])),
             'expert_verified':False,'used_for_training':False}
    write_json(out/'collection.json',summary);print(json.dumps(summary))


def collect(out, per_category=25, max_pages=8, categories=None, exclude_samples=None,
            *, resume=False, max_seconds=None, max_requests=None):
    """Resume one frozen protocol from an atomic cursor, with verified cached pages."""
    from kazstyle.data.checkpoint import CollectionCheckpoint, CollectionPaused
    out = Path(out)
    categories = list(categories or CATEGORIES)
    if not 1 <= per_category <= 1500 or not 1 <= max_pages <= 400:
        raise ValueError('Limits outside allowed range')
    if not 1 <= len(categories) <= 8 or len(set(categories)) != len(categories):
        raise ValueError('Choose 1..8 distinct categories')
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError('Time budget must be positive')
    if max_requests is not None and (type(max_requests) is not int or max_requests < 1):
        raise ValueError('Request budget must be a positive integer')
    excluded = set()
    if exclude_samples:
        for row in json.loads(Path(exclude_samples).read_text(encoding='utf-8')):
            source = row.get('source_url', '')
            if urlsplit(source).netloc == 'surak.baribar.kz':
                match = re.search(r'/(\d+)/?', urlsplit(source).path)
                if match:
                    excluded.add(match[1])
    protocol = {'version':'forum_checkpoint_v1', 'per_category':per_category,
        'max_pages':max_pages, 'categories':categories, 'excluded_thread_ids':sorted(excluded),
        'code_hashes':{name:file_hash(Path(__file__).with_name(name))
                       for name in ['forum.py','quality.py','corpus.py','checkpoint.py']},
        'unit':'whole question; source-derived style requires semantic review'}
    initial = {'accepted':[], 'rejected':[], 'errors':[], 'seen':[], 'cache':{},
        'category_index':0, 'page_index':0, 'url':None, 'pending':[], 'next_url':None,
        'category_started':False, 'category_links':{}, 'blocked':False}
    job = CollectionCheckpoint(out, protocol, initial, resume)
    state = job.state
    if state['blocked']:
        raise ValueError('Source access/rate limit was recorded; do not automatically retry this collection')
    cache = out/'pages';cache.mkdir(exist_ok=True)
    accepted, rejected, errors = state['accepted'], state['rejected'], state['errors']
    seen = set(state['seen']);hashes = {r['content_hash'] for r in accepted}
    counts = Counter(r['category'] for r in accepted)
    session = requests.Session();agent = 'KazStyleResearch/1.0'
    session.headers['User-Agent'] = agent
    robot = RobotFileParser();started = time.monotonic();last = 0.;requests_made = 0

    def save(status='running'):
        state['seen'] = sorted(seen)
        job.save(status)

    def budget_check():
        if (out/'STOP').exists():
            raise CollectionPaused('stop_file')
        if max_seconds is not None and time.monotonic()-started >= max_seconds:
            raise CollectionPaused('time_budget')
        if max_requests is not None and requests_made >= max_requests:
            raise CollectionPaused('request_budget')

    def fetch(url):
        nonlocal last, requests_made
        budget_check()
        if urlsplit(url).netloc != 'surak.baribar.kz' or not robot.can_fetch(agent,url):
            raise ValueError('Unexpected host or disallowed URL')
        previous = state['cache'].get(url)
        if previous:
            path = cache/previous['filename']
            if file_hash(path) != previous['sha256']:
                raise RuntimeError('Cached source checksum mismatch')
            return BeautifulSoup(path.read_bytes(),'html.parser'), previous['actual_url'], path
        time.sleep(max(0.,1.-(time.monotonic()-last)));last = time.monotonic()
        requests_made += 1
        with session.get(url,timeout=(8,20),stream=True) as response:
            if response.status_code in {401,403,429}:
                state['blocked'] = True
                raise CollectionPaused('source_access_or_rate_limit')
            response.raise_for_status();parts=[];size=0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > 3*1024*1024:
                    raise ValueError('Page byte limit')
                parts.append(chunk)
            raw = b''.join(parts);path = cache/(digest(url)+'.html')
            path.write_bytes(raw)
            state['cache'][url] = {'filename':path.name,'sha256':file_hash(path),'actual_url':response.url}
            # Cursor still points to this URL; interruption here safely re-parses the saved page.
            save()
            return BeautifulSoup(raw,'html.parser'), response.url, path

    stop_reason = None
    try:
        budget_check()
        response = session.get(BASE+'robots.txt',timeout=(8,20))
        if response.status_code in {401,403,429}:
            state['blocked'] = True
            raise CollectionPaused('robots_access_or_rate_limit')
        response.raise_for_status();robot.parse(response.text.splitlines())
        (out/'robots.txt').write_text(response.text,encoding='utf-8')
        if not state['category_links']:
            listing,_,_ = fetch(BASE+'categories')
            state['category_links'] = {
                unquote(urlsplit(urljoin(BASE,a['href'])).path).rsplit('/',1)[-1]:urljoin(BASE,a['href'])
                for a in listing.select('a[href]') if '/questions/' in a['href'] or a['href'].startswith('./questions/')}
            save()
        while state['category_index'] < len(categories):
            budget_check()
            category = categories[state['category_index']]
            if not state['category_started']:
                state['url'] = state['category_links'].get(category)
                state['category_started'] = True
                if not state['url']:
                    errors.append({'category':category,'error':'Category link not found'})
            if counts[category] >= per_category or state['page_index'] >= max_pages or not state['url']:
                state.update(category_index=state['category_index']+1, page_index=0,
                             url=None, pending=[], next_url=None, category_started=False)
                save();continue
            if not state['pending']:
                listing,base,_ = fetch(state['url'])
                state['pending'] = list(dict.fromkeys(urljoin(base,a['href']) for a in listing.select('.qa-q-item-title a[href]')))
                next_link = next((a['href'] for a in listing.select('.qa-page-links a[href]')
                                  if a.get_text(strip=True) in {'»','›','Next'}),None)
                state['next_url'] = urljoin(base,next_link) if next_link else None
                save()
            while state['pending'] and counts[category] < per_category:
                budget_check()
                link = state['pending'][0];match = re.search(r'/(\d+)/?',urlsplit(link).path)
                if match and match[1] not in seen and match[1] not in excluded:
                    try:
                        page,actual,path = fetch(link);raw,author = parse_question(str(page))
                        text,cleaning = clean_text(raw);stats,reasons = inspect_text(text,min_words=20,max_words=500)
                        content_hash = digest(text.casefold())
                        if content_hash in hashes:reasons.append('exact_duplicate')
                        if has_contact_identifier(text):reasons.append('possible_contact_or_identifier')
                        row = {'doc_id':digest('surak_question:'+match[1])[:24],'text':text,'style':'colloquial',
                            'source_url':actual,'source_domain':'surak.baribar.kz','parent_id':'surak_thread:'+match[1],
                            'author_group':author,'genre':'forum_question','category':category,
                            'label_origin':'question_genre_candidate_NOT_verified_style','review_status':'unreviewed',
                            'usage_status':'review_pool_only','content_hash':content_hash,'quality':stats,'cleaning':cleaning,
                            'eligibility_reasons':reasons,'source_snapshot':str(path),'source_sha256':file_hash(path),
                            'collected_at':datetime.now(timezone.utc).isoformat(),
                            'license':'Website attribution notice; no blanket redistribution permission asserted'}
                        (rejected if reasons else accepted).append(row)
                        if not reasons:hashes.add(content_hash);counts[category] += 1
                    except (requests.RequestException,ValueError) as exc:
                        errors.append({'url':link,'error':str(exc)})
                    seen.add(match[1])
                state['pending'].pop(0);save()
            state.update(url=state['next_url'],next_url=None,page_index=state['page_index']+1,pending=[])
            save()
            print(json.dumps({'category':category,'accepted':counts[category],'total':len(accepted),'pages':state['page_index']}),flush=True)
    except CollectionPaused as exc:
        stop_reason = str(exc)
    except (requests.RequestException,ValueError) as exc:
        errors.append({'url':state['url'],'error':str(exc)});stop_reason = 'source_error'
    except KeyboardInterrupt:
        stop_reason = 'interrupted'
    finally:
        session.close()
        save('paused' if stop_reason or state['category_index'] < len(categories) else 'finished')
    # These are derived exports. The atomic checkpoint is authoritative while unfinished.
    write_jsonl(out/'candidates.jsonl',accepted);write_jsonl(out/'quarantine.jsonl',rejected)
    finished = state['category_index'] == len(categories) and not stop_reason
    summary = {'candidates':len(accepted),'quarantined':len(rejected),'by_category':dict(counts),
        'errors':errors,'collection_finished':finished,'stop_reason':stop_reason,
        'network_requests_this_run':requests_made,'excluded_existing_thread_ids':sorted(excluded),
        'protocol':protocol,'expert_verified':False,'used_for_training':False,
        'limitations':['Questions require semantic screening; source genre is not a verified style label.',
                      'Completion means the configured crawl budget was exhausted, not the entire source.']}
    write_json(out/'collection.json',summary)
    if finished:
        write_json(out/'COMPLETE.json',{'candidates_sha256':file_hash(out/'candidates.jsonl'),
                                     'checkpoint_sha256':file_hash(out/'CHECKPOINT.json')})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--per-category',type=int,default=25);p.add_argument('--max-pages',type=int,default=8)
    p.add_argument('--rescreen',type=Path,help='Reprocess a previously collected snapshot without networking')
    p.add_argument('--categories', nargs='+', help='Explicit category slugs verified against the public categories page')
    p.add_argument('--exclude-samples',type=Path,help='Existing/reserved source URLs; skipped before fetching')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--max-seconds',type=float)
    p.add_argument('--max-requests',type=int)
    a=p.parse_args()
    if a.rescreen:rescreen(a.rescreen,a.out_dir)
    else:print(json.dumps(collect(a.out_dir,a.per_category,a.max_pages,a.categories,a.exclude_samples,
                                resume=a.resume,max_seconds=a.max_seconds,max_requests=a.max_requests)))
