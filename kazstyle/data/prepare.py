"""Audit/clean every raw document; keep rejected records with explicit reasons.

This produces a candidate store, not an expert-annotated dataset. Known broken
source labels are excluded by default. Nothing overwrites the raw CSV files.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from kazstyle.data.corpus import STYLES, ALIASES, canonical_url, digest, file_hash, read_csv_strict, write_json, write_jsonl
from kazstyle.data.quality import CLEANING_VERSION, clean_text, inspect_text
from kazstyle.settings import project_path


def prepare(data_dir, out, external=()):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    accepted, rejected, audits = [], [], {}
    seen = {}
    for style in STYLES:
        rows, audits[style] = read_csv_strict(data_dir/f'{style}.csv')
        for row in rows:
            raw_label=row['label'].strip().lower()
            if ALIASES.get(raw_label,raw_label)!=style:
                raise ValueError(f'File/label mismatch in {style}: row {row["source_row"]}')
            text, changes = clean_text(row['text'])
            stats, reasons = inspect_text(text)
            # The original literary collector accepted global navigation categories;
            # the Telegram collector equated channel membership with conversation.
            # Keep these texts for review; never silently relabel them as gold.
            if style in {'literary', 'colloquial'}:
                reasons.append('unreliable_legacy_style_label')
            url = canonical_url(row['source_url'])
            record = dict(doc_id=digest(url)[:24], text=text, style=style,
                          source_url=url, source_domain=urlsplit(url).netloc,
                          source_file=str(data_dir/f'{style}.csv'), source_row=row['source_row'],
                          parent_id=url, label_origin='source_category', review_status='unreviewed',
                          genre={'official':'legal_act','scientific':'educational_article','publicistic':'news'}.get(style,'unknown'),
                          content_hash=digest(text.casefold()), cleaning=changes, quality=stats,
                          eligibility_reasons=reasons)
            (rejected if reasons else accepted).append(record)
        print(f'[clean] {style}: {len(rows)} read', flush=True)
    for path in external:
        audits[str(path)] = {'sha256':file_hash(path)}
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            text, changes = clean_text(row['text'])
            stats, reasons = inspect_text(text, min_words=8)
            if row.get('style') not in STYLES:
                reasons.append('style_unassigned')
            record = {**row, 'text':text, 'content_hash':digest(text.casefold()),
                      'cleaning':changes,'quality':stats,'eligibility_reasons':reasons}
            (rejected if reasons else accepted).append(record)
    # Quarantine every occurrence of an exact-text label conflict.
    labels = {}
    for r in accepted:
        labels.setdefault(r['content_hash'],set()).add(r['style'])
    unique = []
    for r in accepted:
        if len(labels[r['content_hash']]) > 1:
            r['eligibility_reasons'] = ['conflicting_labels']
            rejected.append(r)
        elif r['content_hash'] in seen:
            r['eligibility_reasons'] = ['exact_duplicate']
            r['duplicate_of'] = seen[r['content_hash']]
            rejected.append(r)
        else:
            seen[r['content_hash']] = r['doc_id']
            unique.append(r)
    write_jsonl(out/'candidates.jsonl', unique)
    write_jsonl(out/'quarantine.jsonl', rejected)
    summary = {'cleaning_version':CLEANING_VERSION,'raw_files':audits,
               'candidate_documents':len(unique),'quarantined_records':len(rejected),
               'candidates_by_style':dict(Counter(r['style'] for r in unique)),
               'reasons':dict(Counter(reason for r in rejected for reason in r['eligibility_reasons'])),
               'label_status':'Automatically screened source/category labels; not expert gold',
               'target_documents':22600,'target_met':len(unique)>=22600}
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=True),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=project_path('data'))
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--external',type=Path,nargs='*',default=[])
    a=p.parse_args();prepare(a.data_dir,a.out_dir,a.external)


if __name__=='__main__': main()
