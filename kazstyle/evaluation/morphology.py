"""Compare raw words, Stanza tokenization and lemmas on fixed document splits."""
import argparse
import html
import json
import time
from pathlib import Path

import joblib
import numpy as np

from kazstyle.data.corpus import file_hash, load_manifest, write_json, write_jsonl
from kazstyle.data.morphology import aligned_annotations, project_words
from kazstyle.evaluation.repeated import paired_group_bootstrap
from kazstyle.evaluation.reports import metrics, save_provenance, save_predictions
from kazstyle.settings import PROJECT_ROOT
from kazstyle.training.classical import model_specs


def evaluate(plan_path, annotations, out):
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    protocol = json.loads((annotations/'protocol.json').read_text(encoding='utf-8'))
    summary = json.loads((annotations/'summary.json').read_text(encoding='utf-8'))
    if protocol['plan_sha256'] != file_hash(plan_path):
        raise ValueError('Annotation used a different plan')
    if summary['annotations_sha256'] != file_hash(annotations/'annotations.jsonl'):
        raise ValueError('Annotations changed')
    if plan['representations'] != ['raw', 'stanza_surface', 'stanza_lemma']:
        raise ValueError('Expected all three controls')
    dataset = PROJECT_ROOT/plan['dataset']
    frame, config = load_manifest(dataset)
    if config['manifest_sha256'] != plan['manifest_sha256']:
        raise ValueError('Dataset changed')
    records = [json.loads(line) for line in (annotations/'annotations.jsonl').read_text(encoding='utf-8').splitlines()]
    aligned = aligned_annotations(frame, records)
    inputs = {'raw': frame.text.tolist(),
              'stanza_surface': [project_words(r['words'], 'text') for r in aligned],
              'stanza_lemma': [project_words(r['words'], 'lemma') for r in aligned]}
    out.mkdir(parents=True)
    specs = model_specs(plan['seed'])
    params = {name: {key: str(value) for key, value in specs[name].get_params(deep=True).items()}
              for name in plan['classifiers']}
    write_json(out/'protocol.json', {'plan': plan, 'plan_sha256': file_hash(plan_path),
        'annotations_sha256': file_hash(annotations/'annotations.jsonl'), 'model_parameters': params,
        'bootstrap': {'repeats': 2000, 'seed': 20261006, 'reference': 'stanza_surface within classifier'},
        'created_before_classifier_fit': True, 'selection': 'Report every configuration; no deployment selection'})
    save_provenance(out, dataset, {'plan_sha256': file_hash(plan_path), 'seed': plan['seed'],
        'annotations_sha256': summary['annotations_sha256']})
    write_jsonl(out/'inputs.jsonl', [{'sample_id': row.sample_id, 'split': row.split,
        **{name: values[i] for name, values in inputs.items()}} for i, row in enumerate(frame.itertuples())])
    results = {}; predictions = {}; bootstraps = {}
    masks = {split: frame.split == split for split in ['train', 'validation', 'test']}
    for family in plan['classifiers']:
        predictions[family] = {}
        for representation in plan['representations']:
            name = family+'__'+representation
            pipe = model_specs(plan['seed'])[family]
            texts = np.asarray(inputs[representation], dtype=object)
            started = time.perf_counter()
            pipe.fit(texts[masks['train']], frame.loc[masks['train'], 'label'])
            fit_seconds = time.perf_counter()-started
            pred = {split: pipe.predict(texts[mask]) for split, mask in masks.items() if split != 'train'}
            if representation == 'raw':
                # Check that this environment recreates the already declared baseline.
                for split in ['validation', 'test']:
                    baseline_path = PROJECT_ROOT/plan['baseline_run']/(family+'_'+split+'_predictions.jsonl')
                    expected = {r['sample_id']: r['y_pred'] for r in
                                map(json.loads, baseline_path.read_text(encoding='utf-8').splitlines())}
                    wanted = [expected[s] for s in frame.loc[masks[split], 'sample_id']]
                    if not np.array_equal(wanted, pred[split]):
                        raise ValueError('Raw control did not reproduce the saved baseline')
            bundle = {'pipeline': pipe, 'representation': representation,
                      'manifest_sha256': config['manifest_sha256'],
                      'annotation_protocol': protocol, 'purpose': 'Morphology ablation; requires declared preprocessing'}
            joblib.dump(bundle, out/(name+'.joblib'))
            restored = joblib.load(out/(name+'.joblib'))
            if not np.array_equal(pred['test'], restored['pipeline'].predict(texts[masks['test']])):
                raise ValueError('Serialization changed predictions')
            result = {split: metrics(frame.loc[masks[split], 'label'], pred[split], config) for split in pred}
            result.update(fit_seconds=fit_seconds, vocabulary_size=len(pipe['tfidf'].vocabulary_), reload_verified=True)
            results[name] = result
            for split in pred:
                save_predictions(out/(name+'_'+split+'_predictions.jsonl'), frame[masks[split]], pred[split], config)
            predictions[family][representation] = dict(zip(frame.loc[masks['test'], 'sample_id'], map(int, pred['test'])))
            write_json(out/'results.json', results)
            print(f'{name}: validation={result["validation"]["macro_f1"]:.4f} test={result["test"]["macro_f1"]:.4f}', flush=True)
        bootstraps[family] = paired_group_bootstrap(frame[masks['test']].reset_index(drop=True),
            predictions[family], 'stanza_surface')
    write_json(out/'bootstrap.json', bootstraps)
    write_json(out/'COMPLETE.json', {'plan_sha256': file_hash(plan_path),
        'manifest_sha256': config['manifest_sha256'], 'results_sha256': file_hash(out/'results.json'),
        'inputs_sha256': file_hash(out/'inputs.jsonl'), 'configurations': len(results)})
    rows = []
    for name, r in results.items():
        rows.append(f'<tr><td>{html.escape(name)}</td><td>{r["validation"]["macro_f1"]:.4f}</td>'
                    f'<td>{r["test"]["macro_f1"]:.4f}</td><td>{r["vocabulary_size"]}</td></tr>')
    ci = []
    for family, b in bootstraps.items():
        low, high = b['models']['stanza_lemma']['paired_macro_f1_difference_vs_reference_95_interval']
        delta = results[family+'__stanza_lemma']['test']['macro_f1'] - results[family+'__stanza_surface']['test']['macro_f1']
        ci.append(f'<tr><td>{html.escape(family)}</td><td>{delta:+.4f}</td><td>{low:+.4f} … {high:+.4f}</td></tr>')
    page = '''<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Лемматизация: отдельный эксперимент</title>
<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;color:#21364b}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border:1px solid #ccd8e3;padding:12px;text-align:left}th{background:#e9f2f7}.note{padding:18px;background:#fff2d3}code{font-size:14px}</style></head><body>
<h1>Проверка лемматизации и морфологической обработки</h1>
<p class="note">Дополнительный исследовательский эксперимент на прежнем внутреннем test с предварительными метками. Не является новым внешним тестированием и не меняет основную серию или модели сайта.</p>
<p>Stanza 1.15.0, Kazakh KTB: tokenize, mwt, pos, lemma; варианты nocharlm. Готовые модели выполняются локально на CPU. Записываются слова, предполагаемые леммы, части речи и морфологические признаки. Метки стиля не используются при этой обработке. Качество морфологической разметки на нашем корпусе отдельно не измерено.</p>
<p>Сравниваются три представления: raw — прежний текст; stanza_surface — те же словоформы после сегментации и раскрытия многословных токенов Stanza; stanza_lemma — их предсказанные леммы. Сравнение последних двух выделяет дополнительное влияние нормализации. Иначе эффект смены токенизации можно ошибочно приписать лемматизации. POS и морфологические теги сохранены для аудита, но не подаются этим классификаторам.</p>
<p>У всех вариантов одинаковые 610 документов и разбиения 370/120/120, seed 42, настройки TF-IDF и классификаторов. Словарь и IDF обучаются только на train. Поиск гиперпараметров не выполнялся. Контроль raw воспроизвёл все validation/test-предсказания исходных линейных моделей.</p>
<table><tr><th>Модель и представление</th><th>Validation Macro-F1</th><th>Test Macro-F1</th><th>Размер словаря</th></tr>'''+''.join(rows)+'''</table>
<h2>Леммы по сравнению со словоформами Stanza</h2>
<table><tr><th>Классификатор</th><th>Разница Macro-F1</th><th>95% условный интервал</th></tr>'''+''.join(ci)+'''</table>
<p>2000 парных bootstrap-повторов групп документов внутри классов. Интервалы относятся только к этому набору и seed; они не учитывают смещение источников и не исправлены на множественные сравнения. Лемматизация может убирать полезные для стиля окончания, а предсказатель лемм — ошибаться. Улучшение заранее не предполагается.</p>
<p>В inputs.jsonl сохранён фактический вход каждого варианта. Поле text в файлах предсказаний показывает исходный фрагмент; преобразованный вход восстанавливается по sample_id и имени представления.</p>
<p><a href="results.json">Полные метрики</a> · <a href="bootstrap.json">Интервалы</a> · <a href="protocol.json">Протокол</a> · <a href="https://aclanthology.org/2020.acl-demos.14/">Описание Stanza</a> · <a href="https://universaldependencies.org/treebanks/kk_ktb/index.html">Kazakh KTB</a></p></body></html>'''
    (out/'report.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    evaluate(a.plan, a.annotations, a.out_dir)
