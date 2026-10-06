"""Audit shared passages across candidate pools without invoking a classifier.

Word-shingle containment catches an excerpt inside a longer document, which a
whole-document cosine threshold can miss. Flags require review: quotations and
boilerplate are not automatically labelled plagiarism or duplicate documents.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from kazstyle.data.corpus import digest, file_hash, write_json, write_jsonl

VERSION = 'word_passage_overlap_v2'


def words(text):
    return re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', text).casefold())


def shingles(tokens, width=5):
    return {tuple(tokens[i:i + width]) for i in range(len(tokens) - width + 1)}


def overlap_metrics(shared, left_count, right_count, min_shared=10,
                    containment_threshold=.8, passage_min_shared=40,
                    passage_threshold=.2, short_passage_threshold=.5):
    smaller = min(left_count, right_count)
    containment = shared / smaller if smaller else 0.
    union = left_count + right_count - shared
    result = {'shared_shingles': shared, 'left_shingles': left_count,
              'right_shingles': right_count, 'containment': containment,
              'jaccard': shared / union if union else 0.}
    if shared >= min_shared and containment >= containment_threshold:
        result['flag'] = 'high_containment'
    elif (shared >= passage_min_shared and containment >= passage_threshold) or (shared >= min_shared and containment >= short_passage_threshold):
        result['flag'] = 'shared_passage'
    else:
        return None
    return result


def read_rows(path):
    seen = set()
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            doc_id = row.get('doc_id') or row.get('sample_id')
            if not isinstance(doc_id, str) or not doc_id or doc_id in seen:
                raise ValueError(f'Missing/duplicate document ID: {path}:{line_number}')
            if not isinstance(row.get('text'), str) or not row['text'].strip():
                raise ValueError(f'Missing text: {path}:{line_number}')
            seen.add(doc_id)
            yield row


def identity(row, path):
    return {'doc_id': row.get('doc_id') or row['sample_id'], 'input': str(path),
            'style': row.get('style', row.get('label')), 'split': row.get('split'),
            'parent_id': row.get('parent_id', row.get('parent_group')),
            'source_domain': row.get('source_domain'),
            'text_sha256': digest(row['text'])}


def audit(candidate_paths, reference_paths, out, width=5):
    if out.exists():
        raise FileExistsError(out)
    if not 2 <= width <= 12:
        raise ValueError('Shingle width outside 2..12')
    paths = [*candidate_paths, *reference_paths]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('An input may only appear once')
    protocol = {'version': VERSION, 'width': width, 'normalization': 'NFKC casefold; Unicode letters/digits; punctuation ignored; numbers preserved',
                'min_shared': 10, 'containment_threshold': .8,
                'passage_min_shared': 40, 'passage_threshold': .2,
                'short_passage_threshold': .5,
                'candidates': [{'path': str(p), 'sha256': file_hash(p)} for p in candidate_paths],
                'references': [{'path': str(p), 'sha256': file_hash(p)} for p in reference_paths],
                'classifier_used': False, 'automatic_removal': False,
                'limitations': ['Not semantic similarity or paraphrase detection.',
                    'A flag can be a quotation or boilerplate; review is required.',
                    'No claim of absence of overlap with encoder pretraining.',
                    'Existing corpora and model results are not modified.']}
    out.mkdir(parents=True)
    shutil.copy2(__file__, out / 'code_snapshot.py')
    write_json(out / 'protocol.json', protocol)
    items, index, exact = [], defaultdict(list), defaultdict(list)
    for path in candidate_paths:
        for row in read_rows(path):
            tokens = words(row['text']); grams = shingles(tokens, width)
            i = len(items)
            items.append({'identity': identity(row, path), 'grams': grams,
                          'normalized_hash': digest(' '.join(tokens)),
                          'author_group': row.get('author_group') if isinstance(row.get('author_group'),str) and row['author_group'].strip() else None,
                          'words': len(tokens)})
            for gram in grams:
                index[gram].append(i)
            exact[items[-1]['normalized_hash']].append(i)
    findings = []

    def compare(grams, normalized_hash, right, within_index=None):
        hits = Counter(i for gram in grams for i in index.get(gram, ()))
        for i in exact.get(normalized_hash, ()):
            hits.setdefault(i, 0)
        for i, shared in hits.items():
            if within_index is not None and i >= within_index:
                continue
            left = items[i]
            metrics = overlap_metrics(shared, len(left['grams']), len(grams))
            if left['normalized_hash'] == normalized_hash:
                metrics = {'flag': 'normalized_exact', 'shared_shingles': shared,
                           'left_shingles': len(left['grams']), 'right_shingles': len(grams),
                           'containment': 1., 'jaccard': 1.}
            if metrics:
                findings.append({'left': left['identity'], 'right': right, **metrics,
                                 'comparison': 'candidate_candidate' if within_index is not None else 'candidate_reference'})

    for j, item in enumerate(items):
        compare(item['grams'], item['normalized_hash'], item['identity'], j)
    reference_count = 0
    for path in reference_paths:
        for row in read_rows(path):
            tokens = words(row['text'])
            compare(shingles(tokens, width), digest(' '.join(tokens)), identity(row, path))
            reference_count += 1
            if reference_count % 2000 == 0:
                print(f'overlap: {reference_count} reference documents; {len(findings)} flags', flush=True)
    for entry in protocol['candidates'] + protocol['references']:
        if file_hash(entry['path']) != entry['sha256']:
            raise ValueError('Input changed during audit')
    findings.sort(key=lambda r: (r['comparison'], r['left']['doc_id'], -r['containment'], r['right']['doc_id']))
    write_jsonl(out / 'findings.jsonl', findings)
    by_author = defaultdict(list)
    for item in items:
        if item['author_group']:
            by_author[item['author_group']].append(item['identity'])
    repeated_authors = [{'author_group': k, 'documents': v} for k, v in sorted(by_author.items()) if len(v) > 1]
    write_json(out / 'repeated_authors.json', repeated_authors)
    summary = {'version': VERSION, 'candidate_documents': len(items),
               'reference_documents': reference_count, 'flags': len(findings),
               'by_flag': dict(Counter(r['flag'] for r in findings)),
               'by_comparison': dict(Counter(r['comparison'] for r in findings)),
               'flagged_candidates': len({(r['left']['input'], r['left']['doc_id']) for r in findings}
                    | {(r['right']['input'], r['right']['doc_id']) for r in findings if r['comparison'] == 'candidate_candidate'}),
               'candidate_documents_with_known_author': sum(bool(r['author_group']) for r in items),
               'repeated_author_groups': len(repeated_authors),
               'used_for_training': False, 'expert_verified': False}
    write_json(out / 'summary.json', summary)
    rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in
                  [r['comparison'], r['flag'], r['left']['doc_id'], r['right']['doc_id'],
                   r['shared_shingles'], f"{r['containment']:.3f}", r['left']['style'], r['right']['style']]) + '</tr>' for r in findings)
    page = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Совпадения фрагментов корпуса</title>
<style>body{{font:16px system-ui;margin:40px;max-width:1500px}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ddd;padding:7px}}pre{{white-space:pre-wrap}}</style>
<h1>Совпадения фрагментов корпуса</h1><p>Сравнение текста без классификаторов. Флаг требует проверки: цитата или шаблон не равны дубликату целого документа.</p>
<pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre>
<p><a href="protocol.json">Протокол и SHA-256 входов</a> · <a href="findings.jsonl">Полные пары</a> · <a href="repeated_authors.json">Повторяющиеся авторы</a></p>
<table><tr><th>Сравнение</th><th>Флаг</th><th>Кандидат</th><th>Другой документ</th><th>Общие 5-граммы</th><th>Доля меньшего</th><th>Стиль 1</th><th>Стиль 2</th></tr>{rows}</table>
<p>Не проверяет парафразы, семантические совпадения и пересечение с предобучением encoder. Исходные корпуса и результаты моделей сохранены.</p></html>'''
    (out / 'report.html').write_text(page, encoding='utf-8')
    write_json(out / 'COMPLETE.json', {'summary_sha256': file_hash(out / 'summary.json'),
               'findings_sha256': file_hash(out / 'findings.jsonl'), 'code_sha256': file_hash(__file__)})
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, action='append', required=True)
    parser.add_argument('--reference', type=Path, action='append', default=[])
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    audit(args.candidates, args.reference, args.out_dir)
