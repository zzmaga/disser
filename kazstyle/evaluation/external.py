"""Frozen, independently sourced qualitative probe; never trains or selects models."""
import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from kazstyle.data.corpus import file_hash, load_manifest, make_excerpt, write_json, write_jsonl
from kazstyle.data.quality import clean_text, assert_text_only
from kazstyle.evaluation.reports import metrics
from kazstyle.inference.service import InferenceService
from kazstyle.models.tokenization import fit_shared_text


def canonical_text(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def evaluate(samples_path, candidate_path, out, deployment=None, purpose='exploratory'):
    if out.exists():
        raise FileExistsError(out)
    samples = json.loads(samples_path.read_text(encoding='utf-8'))
    if len({r['id'] for r in samples}) != len(samples):
        raise ValueError('Probe IDs must be unique')
    service = InferenceService(deployment=deployment)
    frame, config = load_manifest(service.dataset)
    labels = {name: int(label) for label, name in config['id_to_label'].items()}
    if any(r['expected'] not in labels for r in samples):
        raise ValueError('Unknown reference label')
    train = frame[frame.split == 'train'].reset_index(drop=True)
    candidate_texts = [(row['doc_id'], canonical_text(row['text'])) for row in
        (json.loads(line) for line in candidate_path.read_text(encoding='utf-8').splitlines())]
    # Diagnostic similarity only; this vectorizer is never attached to a classifier.
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=60000)
    train_vectors = vectorizer.fit_transform(train.text)
    audits = []
    for sample in samples:
        cleaned, _ = clean_text(sample['text'])
        excerpt, _ = make_excerpt(cleaned, service.tokenizer, config['max_words'], config['max_tokens'])
        excerpt=fit_shared_text(excerpt,service.constraints)
        assert_text_only([excerpt])
        normalized = canonical_text(excerpt)
        matches = [doc_id for doc_id, text in candidate_texts if normalized and normalized in text]
        similarities = (vectorizer.transform([excerpt]) @ train_vectors.T).toarray()[0]
        nearest = int(np.argmax(similarities))
        domain = urlsplit(sample['source_url']).netloc
        audits.append({'id': sample['id'], 'exact_substring_in_candidate_pool': matches,
                      'nearest_train_sample': train.iloc[nearest]['sample_id'],
                      'nearest_train_char_cosine': float(similarities[nearest]),
                      'source_domain_in_training': domain in set(train.source_domain),
                      'eligible_for_clean_subset': not matches and float(similarities[nearest]) < .9,
                      'analyzed_text': excerpt})
    out.mkdir(parents=True)
    # Freeze references and eligibility BEFORE any classification.
    write_json(out / 'protocol.json', {
        'purpose': ('Regression on previously inspected examples; not an unseen test' if purpose=='regression'
                    else 'Exploratory web probe, not an expert-labelled or representative benchmark'),
        'label_origin': 'assistant semantic review before model inference',
        'samples_sha256': file_hash(samples_path), 'candidate_pool_sha256': file_hash(candidate_path),
        'manifest_sha256': config['manifest_sha256'], 'counts': dict(Counter(r['expected'] for r in samples)),
        'exclusion_rule': 'Exact analyzed excerpt in candidate pool OR character cosine >=0.90 against train',
        'pretraining_overlap': 'Unknown; classic literary texts may have appeared in encoder pretraining',
        'selection_policy': 'Convenience sample; no re-selection after seeing predictions',
        'overlap_audit': audits})
    predictions = []
    for sample, audit in zip(samples, audits):
        response = service.classify(sample['text'], compare=True)
        if response['excerpt'] != audit['analyzed_text']:
            raise RuntimeError('Inference input differs from frozen overlap audit')
        row = {**sample, 'eligible_for_clean_subset': audit['eligible_for_clean_subset'], 'response': response}
        predictions.append(row)
        print(json.dumps({'id': sample['id'], 'expected': sample['expected'],
                          'eligible': row['eligible_for_clean_subset'],
                          'predictions': {r['model']: r['style'] for r in response['results']}}, ensure_ascii=True), flush=True)
    write_jsonl(out / 'predictions.jsonl', predictions)
    results = {}
    for key in service.models:
        results[key] = {}
        for cohort, rows in [('all_probes', predictions), ('no_detected_overlap', [r for r in predictions if r['eligible_for_clean_subset']])]:
            if not rows:
                continue
            y = [labels[r['expected']] for r in rows]
            p = [labels[next(v['style'] for v in r['response']['results'] if v['model'] == key)] for r in rows]
            results[key][cohort] = {**metrics(y, p, config), 'correct': sum(a == b for a, b in zip(y, p))}
    write_json(out / 'results.json', results)
    columns = ''.join('<th>'+html.escape(v['name'])+'</th>' for v in service.models.values())
    body = []
    for row in predictions:
        cells = ''.join('<td class="'+('ok' if p['style'] == row['expected'] else 'error')+'">'+html.escape(p['style_ru'])+'</td>' for p in row['response']['results'])
        preview = ' '.join(row['text'].split()[:12]) + '…'
        body.append('<tr><td>'+html.escape(row['id'])+'<br><a href="'+html.escape(row['source_url'], quote=True)+'">Источник</a><br>'+html.escape(preview)+'</td><td>'+html.escape(row['expected'])+'</td><td>'+('Нет обнаруженных повторов' if row['eligible_for_clean_subset'] else 'Обнаружено пересечение')+'</td>'+cells+'</tr>')
    scores = ''.join('<li>'+html.escape(service.models[k]['name'])+': '+str(v['all_probes']['correct'])+'/'+str(v['all_probes']['n'])+' совпадений с предварительной меткой.</li>' for k,v in results.items())
    content = '<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка моделей на новых источниках</title><style>body{font:16px/1.5 system-ui;margin:36px;color:#15283b}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccd3da;text-align:left;vertical-align:top}th{background:#eef2f6}.error{background:#ffe5e5}.ok{background:#edf7ed}a{color:#174c90}</style><h1>Проверка моделей на новых источниках</h1><p>Небольшая целевая подборка. Метки определены ассистентом до запуска моделей и требуют экспертного подтверждения. Это диагностика ошибок, а не независимая окончательная оценка диссертации. После просмотра ответов тексты не заменялись, модели не дообучались.</p><p>Проверены точные фрагменты в пуле кандидатов и близость к train. Отсутствие обнаруженных совпадений не доказывает отсутствие всех повторов или знакомства encoder с текстами при предобучении.</p><ul>'+scores+'</ul><table><tr><th>Пример</th><th>Ожидаемый стиль</th><th>Проверка повторов</th>'+columns+'</tr>'+''.join(body)+'</table></html>'
    if purpose=='regression':
        content=content.replace('Проверка моделей на новых источниках','Повторная проверка известных веб-примеров')
        content=content.replace('<p>Небольшая целевая подборка.', '<p><strong>Регрессионная проверка: эти тексты и предыдущие ответы уже были просмотрены при разработке. Это не новый скрытый test.</strong></p><p>Небольшая целевая подборка.')
    (out / 'report.html').write_text(content, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--candidate-pool', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--deployment', type=Path)
    parser.add_argument('--purpose', choices=['exploratory','regression'], default='exploratory')
    args = parser.parse_args()
    evaluate(args.samples, args.candidate_pool, args.out_dir, args.deployment, args.purpose)
