"""Save university form sources; extract DOCX text without executing office files."""
import argparse
import io
import json
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup
from kazstyle.data.corpus import digest, file_hash, write_json, write_jsonl
from kazstyle.data.quality import clean_text, inspect_text

BASE = 'https://students.ayu.edu.kz/'
INDEX = BASE + 'kk/formlar'


def is_docx_package(raw):
    if not raw.startswith(b'PK\x03\x04'):
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            return 'word/document.xml' in archive.namelist()
    except zipfile.BadZipFile:
        return False


def extract_docx(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entry = archive.getinfo('word/document.xml')
        if entry.file_size > 10 * 1024 * 1024:
            raise ValueError('Document XML byte limit')
        xml = archive.read(entry)
    if b'<!DOCTYPE' in xml or b'<!ENTITY' in xml:
        raise ValueError('Unexpected XML declaration')
    root = ElementTree.fromstring(xml)
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    paragraphs = []
    for paragraph in root.findall('.//w:body//w:p', ns):
        paragraphs.append(''.join(t.text or '' for t in paragraph.findall('.//w:t', ns)))
    return '\n'.join(paragraphs)


def collect(out, resume=False):
    if (out/'collection.json').exists() or (out.exists() and not resume):
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True); (out/'files').mkdir(exist_ok=True)
    session = requests.Session(); agent = 'KazStyleResearch/1.0'; session.headers['User-Agent'] = agent
    response = session.get(BASE+'robots.txt', timeout=25)
    if response.status_code not in {200,404}:
        raise ValueError('Robots policy unavailable')
    (out/'robots.txt').write_bytes(response.content)
    robot = RobotFileParser(); robot.parse(response.text.splitlines() if response.status_code==200 else [])
    def fetch(url):
        if urlsplit(url).netloc != 'students.ayu.edu.kz' or not robot.can_fetch(agent,url):
            raise ValueError('Host/robots restriction')
        with session.get(url,timeout=(8,25),stream=True) as r:
            r.raise_for_status(); blocks=[]; size=0
            for chunk in r.iter_content(65536):
                size += len(chunk)
                if size > 2*1024*1024:
                    raise ValueError('Download byte limit')
                blocks.append(chunk)
            return b''.join(blocks)
    page=fetch(INDEX); (out/'index.html').write_bytes(page)
    soup=BeautifulSoup(page,'html.parser')
    links={a['href']:a.find_parent('tr').get_text(' ',strip=True) for a in soup.select('a[href]')
           if a.find_parent('tr') and a['href'].lower().endswith(('.doc','.docx'))}
    candidates,quarantine,errors=[],[],[]
    for url,title in list(links.items())[:25]:
        time.sleep(1.)
        try:
            path=out/'files'/(digest(url)+Path(urlsplit(url).path).suffix)
            raw=path.read_bytes() if path.exists() else fetch(url)
            if not path.exists():path.write_bytes(raw)
            row={'doc_id':digest(url)[:24],'title':title,'source_url':url,'source_domain':'students.ayu.edu.kz',
                 'source_index':INDEX,'parent_id':'ayu_form:'+digest(url)[:24],
                 'template_family_id':'ayu_university_forms','style':'official','genre':'application_form',
                 'source_snapshot':str(path),'source_sha256':file_hash(path),
                 'review_status':'unreviewed','label_origin':'university_form_listing_NOT_expert_review',
                 'usage_status':'reserved_source_NOT_for_training_or_prediction',
                 'collected_at':datetime.now(timezone.utc).isoformat(),
                 'license':'Public university forms; no blanket redistribution permission asserted'}
            if not is_docx_package(raw):
                quarantine.append({**row,'eligibility_reasons':['legacy_doc_requires_validated_extractor']}); continue
            text,cleaning=clean_text(extract_docx(raw));stats,reasons=inspect_text(text,min_words=10,max_words=10000)
            row.update(text=text,content_hash=digest(text.casefold()),cleaning=cleaning,quality=stats,eligibility_reasons=reasons)
            (quarantine if reasons else candidates).append(row)
        except requests.HTTPError as exc:
            errors.append({'url':url,'error':str(exc)})
            if exc.response.status_code in {401,403,429}:
                break
        except (requests.RequestException,ValueError,zipfile.BadZipFile,ElementTree.ParseError) as exc:
            errors.append({'url':url,'error':str(exc)})
    write_jsonl(out/'candidates.jsonl',candidates);write_jsonl(out/'quarantine.jsonl',quarantine)
    summary={'candidates':len(candidates),'quarantined':len(quarantine),'errors':errors,
             'reasons':dict(Counter(x for r in quarantine for x in r['eligibility_reasons'])),
             'source_index_sha256':file_hash(out/'index.html'),'code_sha256':file_hash(__file__),
             'usage':'Reserved source candidates; no model queries or training. Not an annotated test.',
             'legacy_files':'Original binary DOC retained without unreliable strings-based extraction.'}
    write_json(out/'collection.json',summary);print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='Resume an incomplete directory; completed collections cannot be changed')
    args=parser.parse_args()
    collect(args.out_dir,args.resume)
