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


def has_contact_identifier(text):
    """Screen Kazakhstan phone numbers (7/8 prefixes) and possible 12-digit IDs.

    Digit boundaries deliberately also match numbers attached to words. This is
    a conservative review filter, not a claim that every match is personal data.
    """
    return bool(re.search(r'(?<!\d)(?:\+?7|8)(?:[\s()-]*\d){10}(?!\d)|(?<!\d)\d{12}(?!\d)', text))


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


def collect(out,per_category=25,max_pages=8,categories=None):
    if out.exists():raise FileExistsError(out)
    if not 1<=per_category<=100 or not 1<=max_pages<=20:raise ValueError('Limits outside allowed range')
    categories = categories or CATEGORIES
    if not 1 <= len(categories) <= 8 or len(set(categories)) != len(categories):
        raise ValueError('Choose 1..8 distinct categories')
    out.mkdir(parents=True);cache=out/'pages';cache.mkdir()
    s=requests.Session();agent='KazStyleResearch/1.0';s.headers['User-Agent']=agent
    robot=RobotFileParser();response=s.get(BASE+'robots.txt',timeout=25)
    if response.status_code in {401,403,429}:raise ValueError('Access/rate limit on robots.txt; collection stopped')
    if response.status_code==200:robot.parse(response.text.splitlines())
    else:robot.parse([])
    (out/'robots.txt').write_text(response.text,encoding='utf-8')
    last=0.;blocked=False
    def fetch(url):
        nonlocal last,blocked
        if blocked:raise ValueError('Collection stopped after access/rate-limit response')
        if urlsplit(url).netloc!='surak.baribar.kz':raise ValueError('Unexpected host')
        if not robot.can_fetch(agent,url):raise ValueError('Disallowed by robots.txt')
        time.sleep(max(0.,1.-(time.monotonic()-last)));last=time.monotonic()
        with s.get(url,timeout=(8,20),stream=True) as r:
            if r.status_code in {401,403,429}:blocked=True
            r.raise_for_status();parts=[];size=0
            for chunk in r.iter_content(65536):
                size+=len(chunk)
                if size>3*1024*1024:raise ValueError('Page byte limit')
                parts.append(chunk)
            raw=b''.join(parts);p=cache/(digest(url)+'.html');p.write_bytes(raw)
            return BeautifulSoup(raw,'html.parser'),r.url,p
    accepted,rejected,errors=[],[],[];seen=set();hashes=set()
    listing,_,_=fetch(BASE+'categories')
    category_links={unquote(urlsplit(urljoin(BASE,a['href'])).path).rsplit('/',1)[-1]:urljoin(BASE,a['href'])
                    for a in listing.select('a[href]') if '/questions/' in a['href'] or a['href'].startswith('./questions/')}
    for category in categories:
        url=category_links.get(category);n=0
        if not url:errors.append({'category':category,'error':'Category link not found'});continue
        for _ in range(max_pages):
            if n>=per_category or not url or blocked:break
            try:
                listing,base,_=fetch(url)
                links=list(dict.fromkeys(urljoin(base,a['href']) for a in listing.select('.qa-q-item-title a[href]')))
                for link in links:
                    if n>=per_category or blocked:break
                    match=re.search(r'/(\d+)/?',urlsplit(link).path)
                    if not match or match[1] in seen:continue
                    seen.add(match[1])
                    try:
                        page,actual,path=fetch(link);raw,author_group=parse_question(str(page))
                        text,cleaning=clean_text(raw);stats,reasons=inspect_text(text,min_words=20,max_words=500)
                        content_hash=digest(text.casefold())
                        if content_hash in hashes:reasons.append('exact_duplicate')
                        if has_contact_identifier(text):reasons.append('possible_contact_or_identifier')
                        row={'doc_id':digest('surak_question:'+match[1])[:24],'text':text,'style':'colloquial',
                             'source_url':actual,'source_domain':'surak.baribar.kz','parent_id':'surak_thread:'+match[1],
                             'author_group':author_group,'genre':'forum_question','category':category,
                             'label_origin':'question_genre_candidate_NOT_verified_style','review_status':'unreviewed',
                             'usage_status':'review_pool_only','content_hash':content_hash,'quality':stats,'cleaning':cleaning,
                             'eligibility_reasons':reasons,'source_snapshot':str(path),'source_sha256':file_hash(path),
                             'collected_at':datetime.now(timezone.utc).isoformat(),
                             'license':'Website attribution notice; no blanket redistribution permission asserted'}
                        (rejected if reasons else accepted).append(row)
                        if not reasons:hashes.add(content_hash);n+=1
                    except (requests.RequestException,ValueError) as exc:errors.append({'url':link,'error':str(exc)})
                next_link=next((a['href'] for a in listing.select('.qa-page-links a[href]') if a.get_text(strip=True) in {'»','›','Next'}),None)
                url=urljoin(base,next_link) if next_link else None
                write_jsonl(out/'candidates.jsonl',accepted);write_jsonl(out/'quarantine.jsonl',rejected)
                print(json.dumps({'forum_category':category,'candidates_in_category':n,'total':len(accepted)},ensure_ascii=True),flush=True)
            except (requests.RequestException,ValueError) as exc:errors.append({'url':url,'error':str(exc)});break
    write_jsonl(out/'candidates.jsonl',accepted);write_jsonl(out/'quarantine.jsonl',rejected)
    summary={'candidates':len(accepted),'categories':categories,'by_category':dict(Counter(r['category'] for r in accepted)),
             'quarantined':len(rejected),'quarantine_reasons':dict(Counter(reason for r in rejected for reason in r['eligibility_reasons'])),
             'errors':errors,'expert_verified':False,'used_for_training':False,
             'limitations':['Questions may contain quoted homework, news or lyrics; genre is not a gold style label.',
                            'One publisher; not a multi-source final test. Thread IDs are available; some author groups are unknown.']}
    write_json(out/'collection.json',summary);print(json.dumps(summary,ensure_ascii=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--per-category',type=int,default=25);p.add_argument('--max-pages',type=int,default=8)
    p.add_argument('--rescreen',type=Path,help='Reprocess a previously collected snapshot without networking')
    p.add_argument('--categories', nargs='+', help='Explicit category slugs verified against the public categories page')
    a=p.parse_args()
    if a.rescreen:rescreen(a.rescreen,a.out_dir)
    else:collect(a.out_dir,a.per_category,a.max_pages,a.categories)
