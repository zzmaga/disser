"""Shared evaluation and experiment provenance for all pilot models."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from markupsafe import Markup

from kazstyle.evaluation.rendering import render_template

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
    for name in ['numpy','pandas','scikit-learn','torch','transformers','tokenizers','joblib','Jinja2']:
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
             PROJECT_ROOT/'configs/requirements-web.txt',
             *sorted((PROJECT_ROOT/'kazstyle').rglob('*.py')),
             *sorted((PROJECT_ROOT/'kazstyle').rglob('*.html'))]
    return {'created_at_utc':datetime.now(timezone.utc).isoformat(), 'python':sys.version,
            'platform':platform.platform(), 'packages':packages, 'git_commit':commit,
            'git_status':status, 'command':sys.argv, 'parameters':params,
            'dataset':str(dataset), 'manifest_sha256':file_hash(dataset/'manifest.jsonl'),
            'source_sha256':{p.relative_to(PROJECT_ROOT).as_posix():file_hash(p) for p in paths if p.exists()}}


def save_predictions(path, part, predictions, config):
    result = []
    for row, pred in zip(part.to_dict('records'), predictions):
        result.append({'sample_id':row['sample_id'], 'doc_id':row['doc_id'],
                       **({'split_group_id':row['split_group_id']} if 'split_group_id' in row else {}),
                       'duplicate_group_id':row['duplicate_group_id'], 'source_url':row['source_url'],
                       'y_true':int(row['label']), 'y_pred':int(pred),
                       'true_name':config['id_to_label'][str(row['label'])],
                       'predicted_name':config['id_to_label'][str(int(pred))],
                       'text':row['text']})
    write_jsonl(path,result)


def save_provenance(out, dataset, params):
    record = provenance(dataset, params)
    for relative, expected in record['source_sha256'].items():
        target = out / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative, target)
        if file_hash(target) != expected:
            raise RuntimeError('Source changed while taking the experiment snapshot')
    write_json(out / 'provenance.json', record)


def render_report(out, frame, config, results, title='Pilot comparison'):
    """Self-contained HTML, no plotting dependency; actual counts and metrics only."""
    names = [config['id_to_label'][str(i)] for i in range(len(config['styles']))]
    counts = frame.groupby(['split','style_name']).size().unstack(fill_value=0)
    limitations = config.get('limitations', [
        'Source labels have not been expert-reviewed; source and class are confounded.',
        'This internal test does not establish generalization to new publishers.'
    ])
    content = render_template('templates/experiment.html', title=title, names=names,
                              limitations=limitations, counts=Markup(counts.to_html()),
                              results=results, manifest_sha256=config['manifest_sha256'])
    (out/'report.html').write_text(content,encoding='utf-8')
    lines = ['# Предварительное сравнение моделей','',
             'Один фрагмент на документ. Ограничения: ' + ' '.join(limitations), '',
             '| Model | Validation Macro-F1 | Test Macro-F1 | Test accuracy | Fit seconds |',
             '|---|---:|---:|---:|---:|']
    for name,r in results.items():
        lines.append(f"| {name} | {r['validation']['macro_f1']:.4f} | {r['test']['macro_f1']:.4f} | {r['test']['accuracy']:.4f} | {r['fit_seconds']:.2f} |")
    lines.extend(['','Dataset SHA-256: '+config['manifest_sha256'], '',
                  'Показаны результаты внутреннего разбиения. Проверки новых источников и известных ошибок сохраняются в отдельных отчётах.'])
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
