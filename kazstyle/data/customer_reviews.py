"""Import a pinned KazSAnDRA training source as unreviewed style candidates.

Sentiment scores remain metadata. No score is interpreted as a style label;
each candidate requires an explicit style review before corpus export.
"""
import argparse
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
import shutil
import zipfile

from kazstyle.data.corpus import digest, file_hash, write_json, write_jsonl
from kazstyle.data.forum import has_contact_identifier
from kazstyle.data.quality import clean_text, inspect_text

DOMAINS = {'appstore', 'bookstore', 'mapping', 'market'}
FIELDS = {'custom_id', 'text', 'text_cleaned', 'label', 'domain'}


def read_source(path):
    """Read one bounded CSV member without extracting paths from a ZIP."""
    with zipfile.ZipFile(path) as archive:
        members = [x for x in archive.infolist() if x.filename.endswith('.csv')]
        if len(members) != 1 or members[0].file_size > 100_000_000:
            raise ValueError('Expected one bounded source CSV')
        with io.TextIOWrapper(archive.open(members[0]), encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or []) != FIELDS:
                raise ValueError('Unexpected KazSAnDRA schema')
            for number, row in enumerate(reader, 2):
                if (None in row or any(value is None for value in row.values())
                        or not row['custom_id'] or row['domain'] not in DOMAINS
                        or row['label'] not in {'0','1','2','3','4'}):
                    raise ValueError(f'Unexpected source record {number}')
                yield number, row


def build(source_dir, out, per_domain=500, seed=42):
    source_dir, out = Path(source_dir), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if not 1 <= per_domain <= 2000:
        raise ValueError('Choose 1..2000 candidates per source domain')
    download = json.loads((source_dir/'DOWNLOAD_COMPLETE.json').read_text(encoding='utf8'))
    if download.get('upstream_split') != 'score_classification_train_imbalanced':
        raise ValueError('Only the original, non-oversampled training source is allowed')
    path = source_dir/download['filename']
    if file_hash(path) != download['sha256']:
        raise ValueError('Source hash mismatch')
    pools, audit, ids, contents = defaultdict(list), [], set(), set()
    counts = Counter()
    for number, original in read_source(path):
        counts['parsed'] += 1
        ident = original['custom_id']
        if ident in ids:
            raise ValueError(f'Repeated upstream ID {ident}; possible resampled input')
        ids.add(ident)
        text, cleaning = clean_text(original['text'])
        content_hash = digest(' '.join(text.casefold().split()))
        words = len(text.split())
        reasons, stats = [], {'words': words}
        if not 12 <= words <= 250:
            reasons.append('length')
        else:
            stats, reasons = inspect_text(text, min_words=12, max_words=250)
            if has_contact_identifier(text):
                reasons.append('possible_contact_or_identifier')
        if content_hash in contents:
            reasons.append('normalized_exact_duplicate')
        contents.add(content_hash)
        locator = {'upstream_custom_id':ident, 'csv_record_number':number,
                   'raw_text_sha256':digest(original['text'])}
        if reasons:
            audit.append(locator | {'reasons': sorted(set(reasons))})
            continue
        row = locator | {
            'doc_id':digest('kazsandra|'+ident)[:24], 'text':text,
            'style':'colloquial', 'genre':'customer_review',
            'parent_id':'kazsandra|'+ident, 'source_domain':'github.com',
            'source_family':'KazSAnDRA', 'upstream_domain':original['domain'],
            'upstream_sentiment_score_zero_based':int(original['label']),
            'upstream_split':download['upstream_split'],
            'source_url':download['url'], 'source_snapshot':path.as_posix(),
            'source_sha256':download['sha256'], 'upstream_revision':download['revision'],
            'label_origin':'candidate_customer_review_genre_NOT_a_style_annotation',
            'review_status':'unreviewed', 'usage_status':'review_pool_only',
            'requires_explicit_style_review':True,
            'author_and_product_ids_available':False,
            'license':download['license'], 'attribution':download['attribution'],
            'content_hash':content_hash, 'quality':stats, 'cleaning':cleaning,
            'eligibility_reasons':[],
        }
        pools[original['domain']].append(row)
    selected = []
    for domain in sorted(pools):
        pool = sorted(pools[domain], key=lambda r:digest(str(seed)+'|'+r['doc_id']))
        selected.extend(pool[:per_domain])
        audit.extend({k:r[k] for k in ['upstream_custom_id','csv_record_number','raw_text_sha256']}
                     | {'reasons':['not_selected_by_domain_budget']} for r in pool[per_domain:])
    report = {'parsed':counts['parsed'], 'eligible_by_domain':{k:len(v) for k,v in pools.items()},
              'selected_by_domain':dict(Counter(r['upstream_domain'] for r in selected)),
              'selected':len(selected), 'audit_reasons':dict(Counter(x for r in audit for x in r['reasons'])),
              'seed':seed, 'per_domain':per_domain, 'expert_verified':False, 'used_for_training':False,
              'limitations':['Sentiment labels do not supply style labels; explicit review required.',
                  'Source CSV lacks original URL, author and product identifiers; do not invent groups.',
                  'Unknown author/product overlap prevents an independent author-held-out evaluation.',
                  'Only upstream training source imported; no upstream validation/test accessed.']}
    out.mkdir(parents=True)
    write_jsonl(out/'candidates.jsonl',selected)
    write_jsonl(out/'audit.jsonl',audit)
    write_json(out/'collection.json',report)
    shutil.copyfile(__file__,out/'code_snapshot.py')
    write_json(out/'COMPLETE.json',{'files':{name:file_hash(out/name) for name in
        ['candidates.jsonl','audit.jsonl','collection.json','code_snapshot.py']},
        'source_sha256':download['sha256']})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--per-domain', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.source_dir,args.out_dir,args.per_domain,args.seed),ensure_ascii=False))
