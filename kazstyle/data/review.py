"""Apply explicit, hash-bound review decisions without silently accepting missing labels."""
import argparse
import json
from collections import Counter
from pathlib import Path

from kazstyle.data.corpus import digest, file_hash, write_json, write_jsonl
from kazstyle.data.quality import inspect_text


def apply_review(candidates, review_file, out):
    if out.exists():
        raise FileExistsError(out)
    review = json.loads(review_file.read_text(encoding='utf-8'))
    if file_hash(candidates) != review['input_sha256']:
        raise ValueError('Review belongs to a different candidate snapshot')
    rows = [json.loads(line) for line in candidates.read_text(encoding='utf-8').splitlines()]
    if len({r['doc_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate candidate IDs')
    version = review.get('schema_version', 1)
    if version not in {1, 2}:
        raise ValueError('Unsupported review schema')
    if version == 2 and (review.get('scope') != 'all_rows' or review.get('reviewer_kind') != 'assistant'):
        raise ValueError('Schema 2 requires all_rows and an explicit assistant reviewer; human reviews use annotate/adjudicate')
    decisions = {r['doc_id']: r for r in review['decisions']}
    if len(decisions) != len(review['decisions']):
        raise ValueError('Duplicate review decisions')
    reviewed_ids = {r['doc_id'] for r in rows if version == 2 or r['style'] == 'colloquial'}
    if reviewed_ids != set(decisions):
        raise ValueError('Every candidate in scope requires exactly one review decision')
    accepted, rejected = [], []
    for row in rows:
        if row['doc_id'] not in reviewed_ids:
            accepted.append(row)
            continue
        decision = decisions[row['doc_id']]
        if type(decision['keep']) is not bool:
            raise ValueError('Review decisions must contain boolean keep values')
        if version == 2:
            if decision.get('text_sha256') != digest(row['text']):
                raise ValueError('Review text hash mismatch')
            if not isinstance(decision.get('reason'), str) or not decision['reason'].strip():
                raise ValueError('A reason is required for each decision')
        row = {**row, 'review_decision': decision, 'review_file_sha256': file_hash(review_file)}
        if decision['keep']:
            if 'text_span' in decision:
                start, end = decision['text_span']
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(row['text']):
                    raise ValueError('Invalid reviewed text span')
                row['upstream_text_sha256'] = digest(row['text'])
                row['text'] = row['text'][start:end]
                row['content_hash'] = digest(row['text'].casefold())
                row['quality'], row['eligibility_reasons'] = inspect_text(row['text'], min_words=8)
                if row['eligibility_reasons']:
                    raise ValueError('Selected review span fails text quality checks')
            row['review_status'] = ('assistant_reviewed_not_expert' if version == 1 or
                decision.get('reviewed_extent') in {'entire_candidate_text', 'selected_span'} else 'assistant_screened_not_expert')
            row['label_origin'] = 'source_plus_assistant_semantic_review'
            row['usage_status'] = 'review_pool_only'
            if decision.get('template_family_id'):
                row['template_family_id'] = decision['template_family_id']
            accepted.append(row)
        else:
            rejected.append({**row, 'review_decision': decision})
    out.mkdir(parents=True)
    write_jsonl(out / 'candidates.jsonl', accepted)
    write_jsonl(out / 'rejected.jsonl', rejected)
    summary = {'input_sha256': file_hash(candidates), 'review_sha256': file_hash(review_file),
               'accepted_by_style': dict(Counter(r['style'] for r in accepted)),
               'rejected': len(rejected), 'expert_verified': False,
               'reviewer': review['reviewer']}
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    apply_review(args.candidates, args.review, args.out_dir)
