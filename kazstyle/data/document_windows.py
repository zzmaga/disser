"""Plan contiguous, shared-tokenizer windows without inventing new documents.

This module prepares development evidence only. It does not assign splits,
train models, aggregate predictions or change the deployed excerpt policy.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import time

from kazstyle.data.corpus import digest, file_hash, write_json, write_jsonl

VERSION = 'whole_document_window_plan_v1'


def partition_document(text, constraints, max_words=160, max_windows=None):
    """Return exact source spans; every complete word occurs at most once.

    constraints consists of (tokenizer_name, tokenizer, token_limit) tuples.
    A bounded preview reports its unprocessed suffix explicitly. A word that
    cannot fit any window is an error, never silently removed or truncated.
    """
    if type(max_words) is not int or max_words < 1:
        raise ValueError('max_words must be a positive integer')
    if max_windows is not None and (type(max_windows) is not int or max_windows < 1):
        raise ValueError('max_windows must be a positive integer or None')
    if (not constraints or len({name for name, _, _ in constraints}) != len(constraints)
            or any(type(limit) is not int or limit < 1 for _, _, limit in constraints)):
        raise ValueError('Unique tokenizer names and positive limits are required')
    words = list(re.finditer(r'\S+', text))
    windows, start = [], 0
    while start < len(words) and (max_windows is None or len(windows) < max_windows):
        end = min(start + max_words, len(words))
        while end > start:
            char_start, char_end = words[start].start(), words[end-1].end()
            part = text[char_start:char_end]
            counts = {name: len(tok(part, add_special_tokens=True, truncation=False,
                                    verbose=False)['input_ids']) for name, tok, _ in constraints}
            if all(counts[name] <= limit for name, _, limit in constraints):
                break
            end -= 1
        if end == start:
            raise ValueError(f'No complete word fits all tokenizers at word {start}, char {words[start].start()}')
        windows.append({'text': part, 'char_start': char_start, 'char_end': char_end,
                        'word_start': start, 'word_end': end, 'words': end-start,
                        'token_counts': counts, 'short_tail_requires_review': end-start < 8})
        start = end
    return {'windows': windows, 'document_words': len(words), 'covered_words': start,
            'word_coverage': start/len(words) if words else 0.0,
            'complete': start == len(words),
            'next_char_start': words[start].start() if start < len(words) else None}


def build(corpus, dataset, selection_file, out):
    """Freeze windows for explicitly selected development documents."""
    from kazstyle.data.export_corpus import load_documents
    from kazstyle.models.tokenization import tokenizer_specs, load_dataset_tokenizer

    corpus, dataset, selection_file, out = map(Path, (corpus, dataset, selection_file, out))
    if out.exists():
        raise FileExistsError('Choose a new window-plan directory')
    selection_sha = file_hash(selection_file)
    selection = json.loads(selection_file.read_text(encoding='utf-8'))
    if selection.get('purpose') != 'development_window_diagnostic':
        raise ValueError('This command accepts development selections only')
    if file_hash(corpus/'COMPLETE.json') != selection['corpus_complete_sha256']:
        raise ValueError('Selection belongs to another corpus snapshot')
    ids = selection['doc_ids']
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Selection needs nonempty unique document IDs')
    rows = {r['doc_id']:r for r in load_documents(corpus)}
    if not set(ids).issubset(rows):
        raise ValueError('Unknown selected document ID')
    config_sha = file_hash(dataset/'config.json')
    config = json.loads((dataset/'config.json').read_text(encoding='utf-8'))
    specs = tokenizer_specs(config)
    loaded = [load_dataset_tokenizer(dataset, config, s['model_name']) for s in specs]
    constraints = [(s['model_name'], tok, s['max_tokens']) for tok, s in loaded]
    windows, documents, failures = [], [], []
    tick = time.monotonic()
    for index, doc_id in enumerate(ids, 1):
        row = rows[doc_id]
        if str(row.get('usage_status', '')).startswith('reserved'):
            raise ValueError('Reserved documents cannot enter this development diagnostic')
        parent_sha = digest(row['text'])
        try:
            planned = partition_document(row['text'], constraints, config['max_words'])
        except ValueError as exc:
            failures.append({'doc_id':doc_id, 'reason':str(exc)})
            continue
        parts = planned.pop('windows')
        for ordinal, window in enumerate(parts):
            windows.append({**window, 'window_id':digest(f'{doc_id}|{parent_sha}|{window["char_start"]}|{window["char_end"]}')[:24],
                'doc_id':doc_id, 'parent_text_sha256':parent_sha, 'window_index':ordinal,
                'split_group_id':row['split_group_id'], 'style':row['style'], 'genre':row.get('genre'),
                'source_url':row.get('source_url'), 'usage_status':'diagnostic_window_plan_only',
                'label_status':'inherited_provisional_document_label_NOT_window_annotation'})
        documents.append({'doc_id':doc_id, 'style':row['style'], 'genre':row.get('genre'),
                          'windows':len(parts), **planned})
        if index % 25 == 0:
            print(f'Window plan: {index}/{len(ids)} documents; {len(windows)} windows', flush=True)
    if (file_hash(selection_file) != selection_sha or file_hash(dataset/'config.json') != config_sha
            or file_hash(corpus/'COMPLETE.json') != selection['corpus_complete_sha256']):
        raise ValueError('Selection or reference dataset changed during processing')
    elapsed = time.monotonic()-tick
    out.mkdir(parents=True)
    write_jsonl(out/'windows.jsonl', windows)
    write_jsonl(out/'documents.jsonl', documents)
    write_jsonl(out/'quarantine.jsonl', failures)
    shutil.copy2(selection_file, out/'selection.json')
    shutil.copy2(__file__, out/'code_snapshot.py')
    summary = {'version':VERSION, 'corpus':str(corpus), 'reference_dataset':str(dataset),
        'reference_config_sha256':config_sha, 'selection_sha256':selection_sha,
        'selected_documents':len(ids), 'processed_documents':len(documents), 'failed_documents':len(failures),
        'windows':len(windows), 'short_windows':sum(w['short_tail_requires_review'] for w in windows),
        'documents_by_style':dict(Counter(d['style'] for d in documents)),
        'all_successful_documents_fully_covered':all(d['complete'] for d in documents),
        'elapsed_seconds':round(elapsed,3), 'tokenizer_limits':{s['model_name']:s['max_tokens'] for s in specs},
        'model_predictions_used':False, 'ready_for_training':False,
        'limitations':['Windows are views of existing documents, not new independent documents.',
            'Document labels do not certify the style of every passage, particularly dialogue and quotations.',
            'Splits must keep entire document/author/work/passage groups together.',
            'Any training must prevent long documents dominating via their number of windows.',
            'This diagnostic does not validate multi-window classification or change the website.']}
    write_json(out/'summary.json', summary)
    write_json(out/'COMPLETE.json', {p.name:file_hash(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    build(args.corpus, args.dataset, args.selection, args.out_dir)
