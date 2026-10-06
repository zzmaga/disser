"""Bounded, pinned downloads of explicitly configured public sources. No auth."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from kazstyle.data.corpus import file_hash, write_json
from kazstyle.settings import project_path


def download(key, out, max_mb=160):
    specs=json.loads(project_path('configs/external_sources.json').read_text(encoding='utf-8'))
    spec=specs[key]
    out.mkdir(parents=True,exist_ok=True)
    target=out/spec['filename']
    metadata=out/'download.json'
    if target.exists():
        saved=json.loads(metadata.read_text(encoding='utf-8'))
        if saved['revision']!=spec['revision'] or saved['sha256']!=file_hash(target):
            raise ValueError('Existing download has changed')
        print(f'[cached] {target}',flush=True);return
    session=requests.Session()
    session.headers['User-Agent']='KazStyleResearch/1.0 (bounded dataset download)'
    card_url=f"https://huggingface.co/datasets/{spec['repo']}/raw/{spec['revision']}/README.md"
    response=session.get(card_url,timeout=(15,40));response.raise_for_status()
    (out/'UPSTREAM_README.md').write_text(response.text,encoding='utf-8')
    url=f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['filename']}"
    size=0
    with session.get(url,stream=True,timeout=(15,60)) as response:
        response.raise_for_status()
        with target.with_suffix(target.suffix+'.part').open('wb') as stream:
            for chunk in response.iter_content(1024*1024):
                size+=len(chunk)
                if size>max_mb*1024*1024:raise ValueError('Download byte budget exceeded')
                stream.write(chunk)
    target.with_suffix(target.suffix+'.part').replace(target)
    write_json(metadata,{**spec,'url':url,'bytes':size,'sha256':file_hash(target),
                         'downloaded_at':datetime.now(timezone.utc).isoformat()})
    print(f'[downloaded] {target}: {size} bytes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=['news','chat'],required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--max-mb',type=int,default=160)
    a=p.parse_args();download(a.source,a.out_dir,a.max_mb)
