"""Shared evaluation and experiment provenance for all pilot models."""
from __future__ import annotations

import html
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from kazstyle.data.corpus import file_hash, write_json, write_jsonl
from kazstyle.settings import PROJECT_ROOT


def metrics(y, predictions, config):
    labels = list(range(len(config['styles'])))
    names = [config['id_to_label'][str(i)] for i in labels]
    return {'n':len(y), 'macro_f1':float(f1_score(y,predictions,labels=labels,average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(y,predictions,labels=labels,average='weighted',zero_division=0)),
            'accuracy':float(accuracy_score(y,predictions)),
            'classification_report':classification_report(y,predictions,labels=labels,target_names=names,
                                                         output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(y,predictions,labels=labels).tolist()}


def provenance(dataset, params):
    packages = {}
    for name in ['numpy','pandas','scikit-learn','torch','transformers','tokenizers','joblib']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        status = subprocess.run(['git','status','--porcelain'],cwd=PROJECT_ROOT,capture_output=True,text=True,check=True).stdout
        commit = subprocess.run(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,capture_output=True,text=True,check=True).stdout.strip()
    except (OSError,subprocess.CalledProcessError):
        status,commit = 'unavailable','unavailable'
    paths = [PROJECT_ROOT/'manage.py', PROJECT_ROOT/'requirements.txt',
             *sorted((PROJECT_ROOT/'kazstyle').rglob('*.py'))]
    return {'created_at_utc':datetime.now(timezone.utc).isoformat(), 'python':sys.version,
            'platform':platform.platform(), 'packages':packages, 'git_commit':commit,
            'git_status':status, 'command':sys.argv, 'parameters':params,
            'dataset':str(dataset), 'manifest_sha256':file_hash(dataset/'manifest.jsonl'),
            'source_sha256':{p.relative_to(PROJECT_ROOT).as_posix():file_hash(p) for p in paths if p.exists()}}


def save_predictions(path, part, predictions, config):
    result = []
    for row, pred in zip(part.to_dict('records'), predictions):
        result.append({'sample_id':row['sample_id'], 'doc_id':row['doc_id'],
                       'duplicate_group_id':row['duplicate_group_id'], 'source_url':row['source_url'],
                       'y_true':int(row['label']), 'y_pred':int(pred),
                       'true_name':config['id_to_label'][str(row['label'])],
                       'predicted_name':config['id_to_label'][str(int(pred))],
                       'text':row['text']})
    write_jsonl(path,result)


def render_report(out, frame, config, results, title='Pilot comparison'):
    """Self-contained HTML, no plotting dependency; actual counts and metrics only."""
    names = [config['id_to_label'][str(i)] for i in range(len(config['styles']))]
    counts = frame.groupby(['split','style_name']).size().unstack(fill_value=0)
    rows, matrices = [], []
    for name, result in results.items():
        te = result['test']
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{result['validation']['macro_f1']:.4f}</td>"
                    f"<td>{te['macro_f1']:.4f}</td><td>{te['accuracy']:.4f}</td>"
                    f"<td>{result['fit_seconds']:.2f}</td></tr>")
        matrix = te['confusion_matrix']
        cells = ''.join('<tr><th>'+html.escape(label)+'</th>'+''.join(f'<td>{v}</td>' for v in vals)+'</tr>'
                        for label,vals in zip(names,matrix))
        matrices.append(f"<h3>{html.escape(name)}</h3><table><tr><th>True / predicted</th>"+
                        ''.join('<th>'+html.escape(n)+'</th>' for n in names)+'</tr>'+cells+'</table>')
    content = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{max-width:1080px;margin:40px auto;padding:0 24px;font:17px/1.5 system-ui;color:#172b42}}
table{{border-collapse:collapse;margin:20px 0;width:100%}}th,td{{border:1px solid #cbd5e1;padding:10px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}th{{background:#eaf0f7}}.note{{background:#fff2cf;padding:18px;border-radius:8px}}</style>
<h1>{html.escape(title)}</h1><p>Три/два стиля, одинаковые тексты и разбиения у всех моделей.
Единица оценки: один фиксированный фрагмент исходного документа.</p>
<p class="note">Предварительный эксперимент. Метки получены из источников и ещё не проверены человеком.
Каждый класс связан с отдельным сайтом. Эти результаты не доказывают перенос на новые источники
и не являются воспроизведением чисел статьи. Test не использовался для настройки.</p>
<h2>Состав выборок</h2>{counts.to_html()}<h2>Результаты</h2>
<table><tr><th>Модель</th><th>Validation Macro-F1</th><th>Test Macro-F1</th><th>Test accuracy</th><th>Обучение, с</th></tr>{''.join(rows)}</table>
<h2>Матрицы ошибок test</h2>{''.join(matrices)}<p>Manifest SHA-256: {config['manifest_sha256']}</p></html>'''
    (out/'report.html').write_text(content,encoding='utf-8')
    lines = ['# Предварительное сравнение моделей','',
             'Метки источников ещё не проверены; источник связан с классом. Один фрагмент на документ.', '',
             '| Model | Validation Macro-F1 | Test Macro-F1 | Test accuracy | Fit seconds |',
             '|---|---:|---:|---:|---:|']
    for name,r in results.items():
        lines.append(f"| {name} | {r['validation']['macro_f1']:.4f} | {r['test']['macro_f1']:.4f} | {r['test']['accuracy']:.4f} | {r['fit_seconds']:.2f} |")
    lines.extend(['','Dataset SHA-256: '+config['manifest_sha256'], '',
                  'Test используется только после фиксации конфигураций. Отдельная проверка новых источников ещё не выполнена.'])
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
