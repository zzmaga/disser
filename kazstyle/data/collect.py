"""Resumable bounded collection from explicit public genre sections.

Only article bodies are retained. Source-category labels are provenance-based,
not human gold. Copyright/licensing is retained per source; no publishing here.
"""
import argparse
import io
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from kazstyle.data.corpus import digest, write_json


def collect(out, per_source=200):
    out.mkdir(parents=True,exist_ok=True)
    cache=out/'pages';cache.mkdir(exist_ok=True)
    records=out/'candidates.jsonl'
    seen={json.loads(line)['doc_id'] for line in records.read_text(encoding='utf-8').splitlines()} if records.exists() else set()
    session=requests.Session();session.headers['User-Agent']='KazStyleResearch/1.0'
    errors=[]
    def fetch(url):
        path=cache/(digest(url)+'.bin')
        if path.exists():return path.read_bytes()
        time.sleep(.4)
        r=session.get(url,timeout=(10,25));r.raise_for_status()
        if len(r.content)>20*1024*1024:raise ValueError('Page budget exceeded')
        path.write_bytes(r.content)
        return r.content
    def save(text,url,style,genre,parent=None,extra=None):
        key=digest(url)[:24]
        if key in seen:return
        row={'doc_id':key,'text':text,'style':style,'source_url':url,
             'parent_id':parent or url,'source_domain':urlsplit(url).netloc,
             'genre':genre,'label_origin':'explicit_source_section','review_status':'unreviewed',
             'collected_at':datetime.now(timezone.utc).isoformat(),
             'license':'Website terms apply; no redistribution permission asserted',**(extra or {})}
        with records.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        seen.add(key)
    sources=[('bilim_folktales','https://bilim-all.kz/article/list?id=4&page={page}', '.list-heading a[href]', r'/article/\d+', '.article-content')]
    for author in [1,3,5,7,9,11,12,14,21,20,19,18,4,15,13,10,8,6]:
        sources.append((f'ertegiler_author_{author}',f'https://ertegiler.kz/author/{author}?page={{page}}',
                        'a[href]',r'/story/','[itemprop="articleBody"]'))
    for name,listing,link_selector,pattern,body_selector in sources:
        visited=set();count=0
        for page in range(1,36):
            try:
                url=listing.format(page=page)
                soup=BeautifulSoup(fetch(url),'html.parser')
                links=sorted({urljoin(url,a['href']) for a in soup.select(link_selector) if re.search(pattern,a['href'])})
                fresh=[u for u in links if u not in visited]
                if not fresh:break
                for link in fresh:
                    visited.add(link)
                    if count>=per_source:break
                    key=digest(link)[:24]
                    if key in seen:count+=1;continue
                    try:
                        article=BeautifulSoup(fetch(link),'html.parser')
                        # Never label recommended/sidebar links as folklore.
                        if name.startswith('ertegiler_author_'):
                            author_id=name.rsplit('_',1)[1]
                            author=article.select_one(f'.primary-content a[href$="/author/{author_id}"]')
                            if author is None:continue
                        body=article.select_one(body_selector)
                        if body is None:raise ValueError('Article body selector missing')
                        for el in body.select('script, style, nav, footer, form, button'):el.decompose()
                        text=body.get_text('\n',strip=True)
                        if len(text.split())<12:continue
                        title=article.find('h1')
                        save(text,link,'literary','folktale',extra={'source_section':name,'title':title.get_text(' ',strip=True) if title else ''})
                        count+=1
                    except requests.HTTPError as e:
                        errors.append({'url':link,'error':str(e)})
                        if e.response.status_code in {403,429}:break
                    except (requests.RequestException,ValueError) as e:errors.append({'url':link,'error':str(e)})
                print(f'[collect] {name} page={page} documents={count}',flush=True)
                if count>=per_source:break
            except (requests.RequestException,ValueError) as e:
                errors.append({'url':url,'error':str(e)});break
    # The following pages were inspected: each is a separate application form.
    pdf_url='https://www.nmu.edu.kz/wp-content/uploads/2017/02/ZHOLSILTEME-ANYKTAMA-08.11.2019.pdf'
    try:
        from pypdf import PdfReader
        reader=PdfReader(io.BytesIO(fetch(pdf_url)))
        for index,page in enumerate(reader.pages):
            text=page.extract_text() or ''
            if 'өтініш үлгісі' not in text.casefold():continue
            # Exclude the explanatory title and printed page number; keep form content.
            text=re.sub(r'^\s*\d+\s*\([^)]*өтініш үлгісі\)\s*','',text,flags=re.I)
            if not re.search(r'сұраймын|өтінемін',text,re.I):continue
            save(text,pdf_url+f'#page={index+1}','official','application_form',parent=pdf_url,
                 extra={'page_number':index+1,'label_origin':'inspected_form_pages'})
    except (requests.RequestException,ValueError) as e:errors.append({'url':pdf_url,'error':str(e)})
    write_json(out/'collection.json',{'documents':len(seen),'errors':errors,'per_source_limit':per_source})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--per-source',type=int,default=200)
    a=p.parse_args();collect(a.out_dir,a.per_source)
