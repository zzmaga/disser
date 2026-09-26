"""Download a bounded, revision-pinned pool for annotation, NEVER labeled training data."""
from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime,timezone
from pathlib import Path
from kazstyle.settings import project_path

import requests

from kazstyle.data.corpus import digest,normalize_text,write_json,write_jsonl

REPO='kz-transformers/multidomain-kazakh-dataset'
REVISION='7a1fcdf9830b1c34b44b3038aafb672447f41890'


class DownloadBudgetExceeded(Exception):
    pass


def bounded_rows(reader,state):
    try:
        yield from reader
    except DownloadBudgetExceeded:
        # csv only yields complete logical records; discard the unfinished last row.
        state['stop_reason']='byte_budget'


class BudgetReader(io.RawIOBase):
    def __init__(self,raw,limit):
        super().__init__();self.raw=raw;self.limit=limit;self.used=0
    def readable(self):return True
    def readinto(self,buffer):
        remaining=self.limit-self.used
        if remaining<=0:raise DownloadBudgetExceeded()
        chunk=self.raw.read(min(len(buffer),remaining),decode_content=True)
        buffer[:len(chunk)]=chunk;self.used+=len(chunk)
        return len(chunk)


def fetch_candidates(source,out,limit=100,max_mb=30):
    if out.exists():raise FileExistsError(f'Refusing overwrite: {out}')
    if not 1<=limit<=2000 or not 1<=max_mb<=200:raise ValueError('Invalid bounded download parameters')
    filename=source+'.csv'
    url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{filename}'
    csv.field_size_limit(2**31-1)
    candidates=[];seen=set();scanned=0;languages={};state={'stop_reason':'end_of_file'}
    with requests.get(url,stream=True,timeout=(20,60)) as response:
        response.raise_for_status()
        budget=BudgetReader(response.raw,max_mb*1024*1024)
        stream=io.TextIOWrapper(io.BufferedReader(budget),encoding='utf-8-sig',newline='')
        reader=csv.DictReader(stream,strict=True)
        columns=reader.fieldnames
        print('Source columns:',columns,flush=True)
        if 'text' not in (columns or []):raise ValueError(f'Unexpected schema: {columns}')
        for index,row in enumerate(bounded_rows(reader,state)):
            scanned+=1
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'Malformed source record {index}')
            text=row['text'];cleaned=normalize_text(text);key=digest(cleaned.casefold())
            lang=row.get('predicted_language','missing')
            languages[lang]=languages.get(lang,0)+1
            if row.get('predicted_language') not in {None,'kaz','kk'}:continue
            if len(cleaned.split())<40 or key in seen:continue
            seen.add(key)
            candidates.append({'candidate_id':f'hf:{source}:{index}', 'text_raw':text,'text_clean':cleaned,
                               'dataset':REPO,'revision':REVISION,'source_file':filename,'row_index':index,
                               'upstream_id':row.get('id'), 'source_url':None,'parent_doc_id':None,
                               'predicted_language':row.get('predicted_language'),'content_hash':key,
                               'style_label':None,'review_status':'unreviewed','reviewer':None,
                               'notes':'Source subset is provenance, not a verified style label. Parent document unknown.'})
            if len(candidates)>=limit:
                state['stop_reason']='requested_count';break
        downloaded=budget.used
    out.mkdir(parents=True)
    write_jsonl(out/'candidates.jsonl',candidates)
    write_json(out/'metadata.json',{'repo':REPO,'revision':REVISION,'source_file':filename,'url':url,
                                  'retrieved_at':datetime.now(timezone.utc).isoformat(),'columns':columns,
                                  'scanned_records':scanned,'retained_records':len(candidates),
                                  'languages_seen':languages,'stop_reason':state['stop_reason'],
                                  'bytes_read':downloaded,'selection':'first qualifying records, NOT a random sample',
                                  'declared_dataset_license':'Apache-2.0',
                                  'usage':'Annotation candidate pool only; not added to training or test.',
                                  'limitations':['No functional-style labels supplied upstream.',
                                                 'Original document/source URL not guaranteed.',
                                                 'Kaz-RoBERTa was pretrained on this dataset; not a pristine external test.']})
    print(json.dumps({'retained':len(candidates),'scanned':scanned,'bytes':downloaded,'output':str(out)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=['kazakhNews','kazakhBooks','cc100-monolingual-crawled-data','leipzig','oscar'],default='kazakhNews')
    p.add_argument('--out-dir',type=Path,default=project_path('data/external/mdbkd_news_candidates'))
    p.add_argument('--limit',type=int,default=100)
    p.add_argument('--max-mb',type=int,default=30)
    a=p.parse_args();fetch_candidates(a.source,a.out_dir,a.limit,a.max_mb)
