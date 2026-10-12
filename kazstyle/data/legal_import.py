"""Screen a pinned public legal corpus; never silently repair malformed records."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import shutil

from kazstyle.data.corpus import digest, file_hash, write_json, write_jsonl
from kazstyle.data.quality import clean_text, inspect_text
from kazstyle.data.forum import has_contact_identifier

VERSION='zan_legal_screen_v1'
QR=re.compile(r'ҚР\s+НҚА\s+ЭББ-гі\s+нақты\s+құжатқа\s+сілтеу\s+QR-коды')
# Exact recurring bilingual PDF furniture observed in the raw source. Do not
# remove arbitrary names or Russian sentences by guessing where they end.
HEADERS=['Правительство Республики Казахстан','(Премьер-Министр РК)',
         'Президент Республики Казахстан']
RUSSIAN_FURNITURE=re.compile(r'\b(?:Маслихат|Акимат|Аким|Министерство|Министерства|Агентство|Республики Казахстан|регулированию|области|района|регистрационной)\b',re.I)

def screen_text(raw):
    removed=[]
    def strip(match):
        removed.append(match.group());return ' '
    text=QR.sub(strip,raw)
    # The repeated publisher header is confined to the first 1,200 characters.
    prefix=text[:1200]
    for header in HEADERS:
        if header in prefix:
            removed.extend([header]*prefix.count(header));prefix=prefix.replace(header,' ')
    text=prefix+text[1200:]
    text,cleaning=clean_text(text)
    stats,reasons=inspect_text(text,min_words=35,max_words=5000)
    if RUSSIAN_FURNITURE.search(text):reasons.append('residual_bilingual_pdf_furniture')
    if re.search(r'QR[- ]|ЭББ|\|.{0,20}\|',text):reasons.append('residual_pdf_furniture_or_table')
    if has_contact_identifier(text):reasons.append('possible_contact_or_identifier')
    # Large numeric/form layouts are not continuous legal prose.
    if sum(c.isdigit() for c in text)/max(1,len(text))>.10:reasons.append('numeric_table_layout')
    if len(re.findall(r'\b[А-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]{1,4}-\s+[А-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]',text))>=8:
        reasons.append('broken_pdf_hyphenation')
    return text,stats,sorted(set(reasons)),{'removed_publisher_furniture':removed,'base':cleaning}

def build(source_dir,out,limit=4500):
    source_dir,out=Path(source_dir),Path(out)
    if out.exists():raise FileExistsError(out)
    if not 1<=limit<=12000:raise ValueError('Choose a bounded new source sample')
    download=json.loads((source_dir/'DOWNLOAD_COMPLETE.json').read_text())
    source=source_dir/'upstream.jsonl.gz'; source_hash=file_hash(source)
    accepted=[];rejected=[];invalid=[];seen={};counts=Counter()
    raw_hash=__import__('hashlib').sha256()
    for number,line in enumerate(gzip.open(source,'rb'),1):
        raw_hash.update(line)
        try:record=json.loads(line)
        except (UnicodeDecodeError,json.JSONDecodeError) as exc:
            invalid.append({'line':number,'raw_sha256':__import__('hashlib').sha256(line).hexdigest(),'bytes':len(line),'error':str(exc)})
            continue
        if record.get('src')!='zan.gov.kz' or not isinstance(record.get('kk'),str):raise ValueError('Upstream schema changed')
        ident=str(record['id']);counts['parsed_documents']+=1
        text,stats,reasons,cleaning=screen_text(record['kk'])
        doc_id=digest('zan_registry_document:'+ident)[:24]
        sha=digest(' '.join(text.casefold().split()))
        if sha in seen:reasons.append('normalized_exact_duplicate')
        else:seen[sha]=doc_id
        row={'doc_id':doc_id,'text':text,'style':'official','genre':'legal_act',
             'parent_id':'zan_registry_document:'+ident,'source_domain':'zan.gov.kz',
             'source_url':'https://huggingface.co/datasets/kurumikz/kazakhstan_legal_corpus',
             'registry_document_id':ident,'upstream_revision':download['revision'],'source_row':number,
             'source_snapshot':source.as_posix(),'source_sha256':source_hash,'source_record_sha256':digest(record['kk']),
             'source_record_locator':'gzip JSONL line; registry ID from upstream, individual URL not supplied',
             'label_origin':'published_legal_register_NOT_expert_annotation','review_status':'unreviewed',
             'usage_status':'review_pool_only','license':'odc-by-1.0','attribution':'kurumikz; source zan.gov.kz',
             'quality':stats,'cleaning':cleaning,'eligibility_reasons':reasons,'cleanup_version':VERSION}
        (rejected if reasons else accepted).append(row)
        if number%5000==0:print('Legal screened',number,'eligible',len(accepted),flush=True)
    if raw_hash.hexdigest()!=download['raw_sha256']:raise ValueError('Raw source changed')
    accepted.sort(key=lambda r:r['doc_id'])
    retained=accepted[:limit]
    out.mkdir(parents=True)
    write_jsonl(out/'candidates.jsonl',retained)
    # No extra full-text copies for thousands of rejected/unselected documents:
    # their exact source remains recoverable from snapshot and line number.
    write_jsonl(out/'audit.jsonl',[{k:v for k,v in r.items() if k not in {'text','cleaning'}} for r in rejected+accepted[limit:]])
    summary={'parsed_documents':counts['parsed_documents'],'eligible_before_sampling':len(accepted),
             'selected':len(retained),'sample_rule':'lowest stable document SHA across entire valid source',
             'not_selected_by_budget':max(0,len(accepted)-limit),'quarantined':len(rejected),
             'invalid_records':invalid,'reasons':dict(Counter(reason for r in rejected for reason in r['eligibility_reasons'])),
             'source_compressed_sha256':source_hash,'source_raw_sha256':raw_hash.hexdigest(),
             'expert_verified':False,'used_for_training':False,
             'limitations':['Historical legal acts, not evidence of current law or everyday business-genre coverage.',
                'Upstream PDF extraction may retain unrecognized noise; no expert style labels.',
                'Malformed upstream record excluded, never repaired into invented text.']}
    write_json(out/'collection.json',summary);shutil.copy2(__file__,out/'code_snapshot.py')
    write_json(out/'COMPLETE.json',{'candidates_sha256':file_hash(out/'candidates.jsonl')})
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--limit',type=int,default=4500)
    a=p.parse_args();print(json.dumps(build(a.source_dir,a.out_dir,a.limit)))
