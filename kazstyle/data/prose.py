"""Bounded prose/story candidates from explicit Adebiportal categories for review."""
import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urljoin,urlsplit,parse_qs
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from kazstyle.data.corpus import digest,file_hash,write_json,write_jsonl
from kazstyle.data.quality import clean_text,inspect_text

BASE='https://adebiportal.kz'
CATEGORIES={'51':'ӘҢГІМЕ','29':'ПРОЗА'}
# Category 41 has the same PROSE display name but mixes memoir, criticism and
# public commentary. It is deliberately excluded after source-section inspection.


def parse_story(raw):
    soup=BeautifulSoup(raw,'html.parser');body=soup.select_one('.content-text');title=soup.find('h1')
    if body is None or title is None:raise ValueError('Story body/title absent')
    category_ids={parse_qs(urlsplit(a['href']).query).get('category',[''])[0]
                  for a in soup.select('a[href]') if '/news/literary?' in a['href']}
    if not category_ids.intersection(CATEGORIES):raise ValueError('Article is not tagged with a selected prose/story category')
    for node in body.select('script,style,nav,form,button,figure'):node.decompose()
    # Separate explicit end credits from narrative; keep them as attribution metadata.
    lines=body.get_text('\n',strip=True).splitlines();credits=[]
    while lines and re.match(r'^(?:Тәржімалаған|Аударған|Қазақ тіліне аударған|Сурет:|Фото:)',lines[-1],re.I):
        credits.insert(0,lines.pop())
    published=soup.find('meta',attrs={'property':'article:published_time'})
    return '\n'.join(lines),{'title':title.get_text(' ',strip=True),'category_ids':sorted(category_ids),
        'credits':credits,'published_at':published.get('content') if published else None}


def rescreen(source,out):
    if out.exists():raise FileExistsError(out)
    accepted=[];rejected=[]
    for name in ['candidates.jsonl','quarantine.jsonl']:
        for row in map(json.loads,(source/name).read_text(encoding='utf-8').splitlines()):
            path=Path(row['source_snapshot'])
            if file_hash(path)!=row['source_sha256']:raise ValueError('Frozen prose source changed')
            if row['source_section'] not in CATEGORIES:
                rejected.append({**row,'eligibility_reasons':['mixed_source_section_requires_individual_genre_review']})
                continue
            raw,meta=parse_story(path.read_bytes());text,cleaning=clean_text(raw)
            stats,reasons=inspect_text(text,min_words=40,max_words=50000)
            record={**row,**meta,'text':text,'quality':stats,'cleaning':cleaning,'eligibility_reasons':reasons}
            (rejected if reasons else accepted).append(record)
    out.mkdir(parents=True)
    write_jsonl(out/'candidates.jsonl',accepted);write_jsonl(out/'quarantine.jsonl',rejected)
    result={'source':str(source),'source_hashes':{name:file_hash(source/name) for name in ['candidates.jsonl','quarantine.jsonl']},
        'candidates':len(accepted),'quarantined':len(rejected),'source_sections':CATEGORIES,
        'decision_basis':'Category 41 mixes criticism, memoir and public commentary despite its PROSE title; excluded as a section, without model predictions.',
        'expert_verified':False,'used_for_training':False,
        'reasons':dict(Counter(reason for r in rejected for reason in r['eligibility_reasons']))}
    write_json(out/'collection.json',result);print(json.dumps(result,ensure_ascii=True))


def collect(out,per_category=30,max_pages=8,exclude_probe=None):
    if out.exists():raise FileExistsError(out)
    if not 1<=per_category<=100 or not 1<=max_pages<=20:raise ValueError('Collection limit outside bounded range')
    out.mkdir(parents=True);cache=out/'pages';cache.mkdir()
    agent='KazStyleResearch/1.0';session=requests.Session();session.headers['User-Agent']=agent
    r=session.get(BASE+'/robots.txt',timeout=25);r.raise_for_status()
    robot=RobotFileParser();robot.parse(r.text.splitlines());(out/'robots.txt').write_text(r.text,encoding='utf-8')
    last=0.;blocked=False
    def fetch(url):
        nonlocal last,blocked
        if blocked:raise ValueError('Collection stopped after access/rate-limit response')
        if urlsplit(url).netloc!='adebiportal.kz' or not robot.can_fetch(agent,url):raise ValueError('URL not allowed')
        time.sleep(max(0.,.8-(time.monotonic()-last)));last=time.monotonic()
        with session.get(url,timeout=(8,25),stream=True) as response:
            if response.status_code in {401,403,429}:blocked=True
            response.raise_for_status();parts=[];size=0
            for chunk in response.iter_content(65536):
                size+=len(chunk)
                if size>3*1024*1024:raise ValueError('Page byte budget exceeded')
                parts.append(chunk)
            raw=b''.join(parts);path=cache/(digest(url)+'.html');path.write_bytes(raw)
            return raw,path
    # Category IDs are observed from this public form; never infer category from a whole host.
    raw,_=fetch(BASE+'/kz/news/literary');form=BeautifulSoup(raw,'html.parser')
    available={option.get('value'):option.get_text(' ',strip=True) for option in form.select('select[name="category"] option')}
    if any(available.get(key)!=title for key,title in CATEGORIES.items()):raise ValueError('Prose category form changed')
    terms,_=fetch(BASE+'/kz/terms');(out/'source_terms.html').write_bytes(terms)
    excluded=set()
    if exclude_probe:
        for sample in json.loads(exclude_probe.read_text(encoding='utf-8')):
            if 'adebiportal.kz' in sample.get('source_url',''):
                match=re.search(r'(?:__|/)(\d+)$',urlsplit(sample['source_url']).path)
                if match:excluded.add(match[1])
    candidates=[];quarantine=[];errors=[];seen=set();hashes=set()
    for category in CATEGORIES:
        url=BASE+'/kz/news/literary?category='+category;count=0
        for _ in range(max_pages):
            if not url or count>=per_category or blocked:break
            try:
                raw,_=fetch(url);listing=BeautifulSoup(raw,'html.parser')
                links=list(dict.fromkeys(urljoin(url,a['href']) for a in listing.select('.news-content a[href]') if '/news/view/' in a['href']))
                for link in links:
                    if count>=per_category or blocked:break
                    match=re.search(r'(?:__|/)(\d+)$',urlsplit(link).path)
                    if not match or match[1] in seen or match[1] in excluded:continue
                    seen.add(match[1])
                    try:
                        page,path=fetch(link);body,meta=parse_story(page)
                        text,cleaning=clean_text(body);stats,reasons=inspect_text(text,min_words=40,max_words=50000)
                        if meta['published_at']:
                            published=datetime.fromisoformat(meta['published_at'])
                            if published.tzinfo is None:published=published.replace(tzinfo=timezone.utc)
                            if published>datetime.now(timezone.utc):reasons.append('future_publication_date')
                        sha=digest(text.casefold())
                        if sha in hashes:reasons.append('exact_duplicate')
                        row={'doc_id':digest('adebiportal_article:'+match[1])[:24],'text':text,'style':'literary',
                            'source_url':link,'source_domain':'adebiportal.kz','parent_id':'adebiportal_article:'+match[1],
                            'genre':'literary_prose_candidate','source_section':category,**meta,
                            'label_origin':'explicit_prose_or_story_section_NOT_expert_review','review_status':'unreviewed',
                            'usage_status':'review_pool_only','requires_work_and_author_group_review':True,
                            'content_hash':sha,'quality':stats,'cleaning':cleaning,'eligibility_reasons':reasons,
                            'source_snapshot':str(path),'source_sha256':file_hash(path),
                            'collected_at':datetime.now(timezone.utc).isoformat(),
                            'license':'Website terms saved; no blanket permission to redistribute entire literary works asserted',
                            'license_url':BASE+'/kz/terms'}
                        (quarantine if reasons else candidates).append(row)
                        if not reasons:count+=1;hashes.add(sha)
                    except (requests.RequestException,ValueError) as exc:errors.append({'url':link,'error':str(exc)})
                pages=[urljoin(url,a['href']) for a in listing.select('a[href]') if 'page=' in a['href'] and 'category='+category in a['href']]
                current=int(parse_qs(urlsplit(url).query).get('page',['1'])[0])
                url=next((u for u in pages if parse_qs(urlsplit(u).query).get('page')==[str(current+1)]),None)
                write_jsonl(out/'candidates.jsonl',candidates);write_jsonl(out/'quarantine.jsonl',quarantine)
                print(json.dumps({'category':category,'page':current,'accepted_in_category':count,'total_candidates':len(candidates)}),flush=True)
            except (requests.RequestException,ValueError) as exc:errors.append({'url':url,'error':str(exc)});break
    result={'candidates':len(candidates),'quarantined':len(quarantine),'errors':errors,
        'per_category':per_category,'max_pages':max_pages,'source_sections':CATEGORIES,
        'reasons':dict(Counter(reason for r in quarantine for reason in r['eligibility_reasons'])),
        'excluded_probe_article_ids':sorted(excluded),'expert_verified':False,'used_for_training':False,
        'limitation':'May include essays, mixed texts and installments of the same work. Review genre, author/work grouping and reuse rights before dataset inclusion.'}
    write_json(out/'collection.json',result);print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--per-category',type=int,default=30);p.add_argument('--max-pages',type=int,default=8)
    p.add_argument('--exclude-probe',type=Path)
    p.add_argument('--rescreen',type=Path,help='Re-screen a frozen collection without new requests')
    a=p.parse_args()
    if a.rescreen:rescreen(a.rescreen,a.out_dir)
    else:collect(a.out_dir,a.per_category,a.max_pages,a.exclude_probe)
