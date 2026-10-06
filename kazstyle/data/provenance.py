"""Apply explicit author/work metadata to a frozen prose pool.

Bylines and pen names are publisher attributions, not verified identities. They
are grouping hints, never additional classifier features or expert style labels.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from kazstyle.data.corpus import digest, file_hash, write_json, write_jsonl
from kazstyle.data.quality import inspect_text


def apply_prose_metadata(source, decisions_path, out):
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads(decisions_path.read_text(encoding='utf-8'))
    if file_hash(source) != plan['input_sha256']:
        raise ValueError('Metadata decisions refer to another snapshot')
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    decisions = {r['doc_id']: r for r in plan['decisions']}
    if len(decisions) != len(plan['decisions']) or set(decisions) != {r['doc_id'] for r in rows}:
        raise ValueError('Exactly one metadata decision per document required')
    result, changes = [], []
    for row in rows:
        decision = decisions[row['doc_id']]
        if digest(row['text']) != decision['text_sha256']:
            raise ValueError('Text changed since metadata review')
        field = decision.get('evidence_field')
        if decision['author_kind'] == 'unknown':
            if not decision.get('reason'):
                raise ValueError('Unknown attribution requires a reason')
            author_group = None
        else:
            if field not in {'title', 'text'} or decision['evidence'] not in row[field]:
                raise ValueError('Author attribution evidence absent')
            if decision['author_kind'] not in {'byline', 'pen_name'} or not decision['author_name'].strip():
                raise ValueError('Invalid author attribution')
            author_key = ' '.join(decision['author_name'].casefold().split())
            namespace = row['source_domain'] if decision['author_kind'] == 'pen_name' else 'published_byline'
            author_group = digest(namespace + '|' + decision['author_kind'] + '|' + author_key)[:24]
        text = row['text']
        removed = []
        prefix = decision.get('remove_prefix')
        if prefix:
            if not text.startswith(prefix):
                raise ValueError('Recorded prefix is not exact')
            text = text[len(prefix):].lstrip(); removed.append({'kind': 'credit_header', 'text': prefix})
        suffix = decision.get('remove_suffix')
        if suffix:
            if not text.endswith(suffix):
                raise ValueError('Recorded footer is not an exact suffix')
            text = text[:-len(suffix)].rstrip(); removed.append({'kind': 'credit_footer', 'text': suffix})
        match = re.match(r'^\((?:Әңгіме|әңгіме)\)\s*', text)
        if match:
            removed.append({'kind': 'genre_heading', 'text': match[0]}); text = text[match.end():]
        if not text:
            raise ValueError('Metadata removal emptied the document')
        quality, reasons = inspect_text(text, min_words=8, max_words=50000)
        if reasons:
            raise ValueError(f'Metadata-cleaned text fails quality checks: {row["doc_id"]}: {reasons}')
        result.append({**row, 'text': text, 'content_hash': digest(text.casefold()),
                       'author_name': decision['author_name'], 'author_kind': decision['author_kind'],
                       'author_group': author_group, 'author_attribution': 'publisher_byline_not_identity_verified',
                       'author_evidence': {'field': field, 'text': decision.get('evidence'), 'reason': decision.get('reason')},
                       'work_group': row['parent_id'],
                       'work_group_status': 'publication_proxy_may_contain_multiple_works',
                       'metadata_review': 'assistant_metadata_review_NOT_style_annotation',
                       'upstream_text_sha256': digest(row['text']),
                       'removed_metadata': removed,
                       'quality': quality, 'eligibility_reasons': reasons,
                       'usage_status': 'review_pool_only'})
        if removed:
            changes.append({'doc_id': row['doc_id'], 'removed': removed})
    out.mkdir(parents=True)
    write_jsonl(out/'candidates.jsonl', result)
    write_json(out/'metadata_changes.json', changes)
    counts = Counter(r['author_group'] for r in result if r['author_group'])
    summary = {'documents': len(result), 'attributed_author_groups': len(counts),
               'repeated_author_groups': sum(n > 1 for n in counts.values()),
               'pen_name_documents': sum(r['author_kind'] == 'pen_name' for r in result),
               'unknown_author_documents': sum(r['author_kind'] == 'unknown' for r in result),
               'metadata_cleaned_documents': len(changes), 'input_sha256': file_hash(source),
               'decisions_sha256': file_hash(decisions_path),
               'output_sha256': file_hash(out/'candidates.jsonl'), 'expert_style_review': False,
               'limitations': ['Same spelling does not establish real identity; aliases can hide shared authors.',
                   'Work grouping is publication-based; cross-publication installments still need review.',
                   'Semantic style of the whole work has not been independently annotated.']}
    write_json(out/'summary.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    apply_prose_metadata(args.source, args.decisions, args.out_dir)
