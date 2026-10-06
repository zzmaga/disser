"""Freeze exact text views for review under the same tokenizer limits as training.

This prepares candidate texts, not model predictions or a final test. Reviews
must concern the actual view; source genre never certifies the excerpt's style.
"""
import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from kazstyle.data.corpus import digest, file_hash, make_excerpt, normalized_hash, write_json, write_jsonl
from kazstyle.data.quality import assert_text_only, inspect_text
from kazstyle.models.tokenization import load_dataset_tokenizer, tokenizer_specs, fit_shared_text

VERSION = 'shared_candidate_views_v2'


def prepare(inputs, reference_dataset, out, view_policy='full_or_central'):
    if out.exists():
        raise FileExistsError(out)
    if view_policy not in {'full_or_central','legacy_mixed'}:
        raise ValueError('Unknown view policy')
    config = json.loads((reference_dataset/'config.json').read_text(encoding='utf-8'))
    specs = tokenizer_specs(config)
    loaded = [load_dataset_tokenizer(reference_dataset, config, spec['model_name']) for spec in specs]
    constraints = [(tokenizer, spec['max_tokens']) for tokenizer, spec in loaded]
    records = defaultdict(list)
    for path in inputs:
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            records[row['doc_id']].append({**row, 'view_input_file': str(path)})
    protocol = {'version': VERSION, 'inputs': {str(p): file_hash(p) for p in inputs},
                'reference_dataset': str(reference_dataset),
                'reference_config_sha256': file_hash(reference_dataset/'config.json'),
                'tokenizers': specs, 'view_policy': view_policy,
                'word_budgets': [160] if view_policy == 'full_or_central' else [40,80,160],
                'word_budget_selection': 'fixed 160; preserve shorter complete texts' if view_policy == 'full_or_central' else 'sha256(doc_id) first 8 hex modulo 3',
                'excerpt': 'central contiguous excerpt, then trailing whole words removed to fit every tokenizer',
                'purpose': 'development_candidate_review_NOT_final_test', 'model_predictions_used': False,
                'expert_verified': False, 'split_assignment': None}
    accepted, rejected, duplicate_records = [], [], []
    seen_text = {}
    for doc_id, versions in records.items():
        if len({(r['style'], digest(r['text'])) for r in versions}) != 1:
            rejected.append({'doc_id': doc_id, 'reason': 'conflicting_versions_of_document',
                             'inputs': [r['view_input_file'] for r in versions]})
            continue
        row = versions[0]
        if len(versions) > 1:
            duplicate_records.append({'doc_id': doc_id, 'same_text_occurrences': len(versions)})
        budget = 160 if view_policy == 'full_or_central' else [40, 80, 160][int(digest(doc_id)[:8], 16) % 3]
        text, details = make_excerpt(row['text'], constraints[0][0], budget, constraints[0][1])
        try:
            text = fit_shared_text(text, constraints)
            assert_text_only([text])
            quality, reasons = inspect_text(text, min_words=8)
        except ValueError as exc:
            rejected.append({'doc_id': doc_id, 'reason': str(exc)}); continue
        if reasons:
            rejected.append({'doc_id': doc_id, 'reason': 'view_quality', 'details': reasons}); continue
        text_hash = digest(text)
        if text_hash in seen_text:
            # No silent preference if identical views have conflicting provisional labels.
            first = seen_text[text_hash]
            if first['style'] != row['style']:
                raise ValueError('Conflicting styles for an identical model view; resolve candidates first')
            rejected.append({'doc_id': doc_id, 'reason': 'identical_view', 'duplicate_of': first['doc_id']}); continue
        seen_text[text_hash] = row
        counts = {spec['model_name']: len(tok(text, add_special_tokens=True, truncation=False)['input_ids']) for tok, spec in loaded}
        accepted.append({**row, 'text': text, 'content_hash': digest(text.casefold()),
            'parent_id': row.get('parent_id') or doc_id, 'view_parent_text_sha256': digest(row['text']),
            'parent_content_hash': normalized_hash(row['text']),
            'duplicate_probe': ' '.join(row['text'].split()[:120] + row['text'].split()[-120:]),
            'view_parent_word_count': len(row['text'].split()), 'quality': quality, 'eligibility_reasons': [],
            'model_view': {'version': VERSION, 'policy': view_policy, 'text_sha256': text_hash, 'word_budget': budget,
                'token_counts': counts, 'char_start': details['char_start'],
                'char_end': details['char_start'] + len(text), 'is_excerpt': text != row['text']},
            'view_review_status': 'exact_view_requires_semantic_review',
            'usage_status': row['usage_status'] if str(row.get('usage_status','')).startswith('reserved') else 'review_pool_only'})
    for path, expected in protocol['inputs'].items():
        if file_hash(path) != expected:
            raise ValueError('Candidate input changed while preparing views')
    out.mkdir(parents=True)
    shutil.copy2(__file__, out/'code_snapshot.py')
    write_json(out/'protocol.json', protocol)
    write_jsonl(out/'candidates.jsonl', accepted)
    write_jsonl(out/'quarantine.jsonl', rejected)
    write_json(out/'duplicate_input_records.json', duplicate_records)
    summary = {'input_records': sum(map(len, records.values())), 'unique_document_ids': len(records),
               'views': len(accepted), 'quarantined': len(rejected),
               'by_style': dict(Counter(r['style'] for r in accepted)),
               'by_source': dict(Counter(r['source_domain'] for r in accepted)),
               'expert_verified': False, 'model_predictions_used': False,
               'candidates_sha256': file_hash(out/'candidates.jsonl')}
    write_json(out/'summary.json', summary)
    write_json(out/'COMPLETE.json', {'candidates_sha256': summary['candidates_sha256']})
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--reference-dataset', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--view-policy', choices=['full_or_central','legacy_mixed'],default='full_or_central')
    args = parser.parse_args()
    prepare(args.input, args.reference_dataset, args.out_dir,args.view_policy)
