"""Collect whole, attributed poems with separate author/work provenance."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urljoin,urlsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from kazstyle.data.corpus import digest,file_hash,write_json,write_jsonl
from kazstyle.data.quality import clean_text,inspect_text

BASE='https://bilim-all.kz'

def parse_poem(raw,url):
    soup=BeautifulSoup(raw,'html.parser');body=soup.select_one('.poem-content')
    title=soup.select_one('h1.heading');author=soup.select_one('.olen-author-box figure h5 a[href^="/figure/"]')
    if body is None or title is None or author is None:
        raise ValueError('Missing poem body, title or explicit author attribution')
    author_name=author.get_text(' ',strip=True)
    if not author_name:raise ValueError('Empty author attribution')
    for el in body.select('script,style,nav,form,button'):el.decompose()
    for el in body.select('br'):el.replace_with('\n')
    text=body.get_text('',strip=False)
    lines=[line.strip() for line in text.splitlines() if line.strip()]
    if len(lines)<4:raise ValueError('Expected a multi-line poem')
    return text,{'title':title.get_text(' ',strip=True),'author_name':author_name,
        'author_url':urljoin(BASE,author['href']),'verse_lines':len(lines),
        'author_group':digest('published_byline|byline|'+' '.join(author_name.casefold().split()))[:24],
        'author_attribution':'publisher_byline_not_identity_verified'}

def collect(out,limit=1800,per_author=60,max_pages=600,resume=False):
    out=Path(out)
    if out.exists() and (not resume or (out/'COMPLETE.json').exists()):raise FileExistsError(out)
    if resume and out.exists():
        old=json.loads((out/'protocol.json').read_text())
        if any(old[k]!=v for k,v in [('limit',limit),('per_author',per_author),('max_pages',max_pages)]):
            raise ValueError('Cannot change collection budget during resume')
    if not 1<=limit<=5000 or not 1<=per_author<=100 or not 1<=max_pages<=1500:raise ValueError('Invalid collection bounds')
    out.mkdir(parents=True,exist_ok=resume);(out/'pages').mkdir(exist_ok=resume)
    session=requests.Session();agent='KazStyleResearch/1.0';session.headers['User-Agent']=agent
    response=session.get(BASE+'/robots.txt',timeout=(8,25));response.raise_for_status()
    robot=RobotFileParser();robot.parse(response.text.splitlines());(out/'robots.txt').write_text(response.text,encoding='utf-8')
    last=0.;blocked=False
    def fetch(url):
        nonlocal last,blocked
        if blocked or urlsplit(url).netloc!='bilim-all.kz' or not robot.can_fetch(agent,url):
            raise ValueError('Stopped or disallowed source URL')
        cached=out/'pages'/(digest(url)+'.html.gz')
        if resume and cached.exists():
            with gzip.open(cached,'rb') as stream:raw=stream.read(3*1024*1024+1)
            if len(raw)>3*1024*1024:raise ValueError('Cached page byte limit')
            return raw,cached
        time.sleep(max(0.,.8-(time.monotonic()-last)));last=time.monotonic()
        with session.get(url,stream=True,timeout=(8,22)) as r:
            if r.status_code in {401,403,429}:blocked=True
            r.raise_for_status();parts=[];size=0
            for chunk in r.iter_content(65536):
                size+=len(chunk)
                if size>3*1024*1024:raise ValueError('Page byte limit')
                parts.append(chunk)
            raw=b''.join(parts);path=out/'pages'/(digest(url)+'.html.gz')
            path.write_bytes(gzip.compress(raw,mtime=0));return raw,path
    # Spread across the public archive, not only the newest author uploads.
    page_order=sorted(range(1,max_pages+1),key=lambda p:digest('poetry_page_seed42|'+str(p)))
    write_json(out/'protocol.json',{'limit':limit,'per_author':per_author,'max_pages':max_pages,
        'page_order':page_order,'interval_seconds':.8,'label_status':'source_derived_not_expert',
        'unit':'one complete attributed poem per publication; never stanza fragments'})
    if resume and (out/'code_snapshot.py').exists() and not (out/'initial_code_snapshot.py').exists():
        shutil.copy2(out/'code_snapshot.py',out/'initial_code_snapshot.py')
    shutil.copy2(__file__,out/'code_snapshot.py')
    rows=[];quarantine=[];errors=[];seen=set();hashes=set();authors=Counter();visited=[]
    for number in page_order:
        if len(rows)>=limit or blocked:break
        url=BASE+'/olen?page='+str(number)
        try:
            raw,_=fetch(url);listing=BeautifulSoup(raw,'html.parser');visited.append(number)
            links=list(dict.fromkeys(urljoin(url,a['href']) for a in listing.select('a[href]')
                if re.fullmatch(r'/olen/\d+[^/]*',urlsplit(urljoin(url,a['href'])).path)))
            for link in links:
                if len(rows)>=limit or blocked:break
                if link in seen:continue
                seen.add(link)
                try:
                    raw,path=fetch(link);body,metadata=parse_poem(raw,link)
                    if authors[metadata['author_group']]>=per_author:continue
                    text,cleaning=clean_text(body);stats,reasons=inspect_text(text,min_words=35,max_words=2500)
                    sha=digest(' '.join(text.casefold().split()))
                    if sha in hashes:reasons.append('normalized_exact_duplicate')
                    row={'doc_id':digest(link)[:24],'text':text,'style':'literary','genre':'poem',
                         'source_url':link,'source_domain':'bilim-all.kz','parent_id':link,
                         'work_group':digest('poem_work|'+metadata['author_group']+'|'+metadata['title'].casefold())[:24],
                         **metadata,'source_snapshot':path.as_posix(),'source_sha256':file_hash(path),
                         'snapshot_compression':'gzip','quality':stats,'eligibility_reasons':reasons,'cleaning':cleaning,
                         'review_status':'unreviewed','label_origin':'explicit_poem_body_with_publisher_author_NOT_expert',
                         'usage_status':'review_pool_only','collected_at':datetime.now(timezone.utc).isoformat(),
                         'license':'Publisher and author rights apply; local research copy, not blanket redistribution'}
                    (quarantine if reasons else rows).append(row)
                    if not reasons:hashes.add(sha);authors[metadata['author_group']]+=1
                except (requests.RequestException,ValueError) as e:errors.append({'url':link,'error':str(e)})
            write_jsonl(out/'candidates.jsonl',rows);write_jsonl(out/'quarantine.jsonl',quarantine)
            print('Poetry:',len(rows),'poems,',len(authors),'authors,',len(visited),'listing pages',flush=True)
        except (requests.RequestException,ValueError) as e:errors.append({'url':url,'error':str(e)})
    result={'candidates':len(rows),'quarantined':len(quarantine),'authors':len(authors),'author_counts':dict(authors),
            'visited_listing_pages':visited,'errors':errors,'blocked':blocked,'expert_verified':False,'used_for_training':False,
            'limitations':['Poetry is a distinct literary genre; balance/report it separately from narrative prose.',
                'Publisher attribution does not verify identity; related editions and author-name variants need review.']}
    write_json(out/'collection.json',result);write_json(out/'COMPLETE.json',{'candidates_sha256':file_hash(out/'candidates.jsonl')})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--limit',type=int,default=1800);p.add_argument('--per-author',type=int,default=60)
    a=p.parse_args();print(json.dumps(collect(a.out_dir,a.limit,a.per_author)))
