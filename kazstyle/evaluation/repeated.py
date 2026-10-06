"""Summarize every predeclared seed and group-aware paired uncertainty."""
import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from kazstyle.data.corpus import load_manifest,write_json,file_hash
from kazstyle.evaluation.reports import metrics
from kazstyle.settings import PROJECT_ROOT, artifact_path


def macro_from_matrix(matrix):
    matrix=np.asarray(matrix)
    numerator=2*np.diagonal(matrix,axis1=-2,axis2=-1)
    denominator=matrix.sum(axis=-1)+matrix.sum(axis=-2)
    return np.divide(numerator,denominator,out=np.zeros_like(numerator,dtype=float),where=denominator!=0).mean(axis=-1)


def paired_group_bootstrap(frame,predictions,reference,repeats=2000,seed=20261006):
    if repeats<100:raise ValueError('At least 100 bootstrap draws required')
    group_key='split_group_id' if 'split_group_id' in frame else 'duplicate_group_id'
    labels=sorted(frame.label.unique());groups=list(frame.groupby(group_key,sort=True))
    if any(part.label.nunique()!=1 for _,part in groups):raise ValueError('Bootstrap groups must be class-homogeneous')
    rng=np.random.default_rng(seed)
    weights=np.zeros((repeats,len(groups)),dtype=int)
    for label in labels:
        indices=[i for i,(_,part) in enumerate(groups) if int(part.label.iloc[0])==label]
        weights[:,indices]=rng.multinomial(len(indices),np.full(len(indices),1/len(indices)),size=repeats)
    draws={};summary={}
    for name,pred in predictions.items():
        matrices=np.zeros((len(groups),len(labels),len(labels)),dtype=int)
        for i,(_,part) in enumerate(groups):
            for row in part.itertuples():matrices[i,int(row.label),int(pred[row.sample_id])]+=1
        simulated=np.einsum('bg,gij->bij',weights,matrices)
        draws[name]=macro_from_matrix(simulated)
        summary[name]={'macro_f1_95_percentile_interval':np.quantile(draws[name],[.025,.975]).tolist()}
    if reference not in draws:raise ValueError('Bootstrap reference model absent')
    for name in draws:
        summary[name]['paired_macro_f1_difference_vs_reference_95_interval']=np.quantile(draws[name]-draws[reference],[.025,.975]).tolist()
    return {'repeats':repeats,'seed':seed,'reference':reference,'groups':len(groups),'test_documents':len(frame),
            'group_key':group_key,
            'scheme':f'Resample {group_key} within each true class, same draws for all models; fixed training seed 42.',
            'limitations':'Conditional exploratory intervals on the development test; no correction for multiple comparisons. They do not measure unseen-source bias or training-seed variation.',
            'models':summary}


def summarize(suite,out):
    if out.exists():raise FileExistsError(out)
    protocol=json.loads((suite/'protocol.json').read_text(encoding='utf-8'))
    status=json.loads((suite/'status.json').read_text(encoding='utf-8'))
    if status['status']!='complete':raise ValueError('Wait for every predeclared run; partial suites cannot be reported as complete')
    plan=protocol['plan'];dataset=PROJECT_ROOT/plan['dataset'];frame,config=load_manifest(dataset)
    if plan['manifest_sha256']!=config['manifest_sha256']:raise ValueError('Wrong dataset')
    test=frame[frame.split=='test'].reset_index(drop=True)
    pooled=defaultdict(list);seed42={};evidence={}
    for run_id in status['completed']:
        run=artifact_path(run_id)
        complete=json.loads((run/'COMPLETE.json').read_text(encoding='utf-8'))
        provenance=json.loads((run/'provenance.json').read_text(encoding='utf-8'))
        if complete['plan_sha256']!=protocol['plan_sha256'] or provenance['manifest_sha256']!=config['manifest_sha256']:
            raise ValueError('Run belongs to a different plan or dataset')
        seed=provenance['parameters']['seed']
        results=json.loads((run/'results.json').read_text(encoding='utf-8'))
        for name,result in results.items():
            path=run/(name+'_test_predictions.jsonl' if 'classical' in run_id else 'test_predictions.jsonl')
            records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            indexed={r['sample_id']:r for r in records}
            if len(indexed)!=len(records) or set(indexed)!=set(test.sample_id):raise ValueError('Predictions missing/duplicated')
            for row in test.itertuples():
                r=indexed[row.sample_id]
                if r['y_true']!=row.label or r['text']!=row.text:raise ValueError('Prediction record differs from frozen text/label')
            prediction={key:r['y_pred'] for key,r in indexed.items()}
            actual=metrics(test.label,[prediction[s] for s in test.sample_id],config)
            if abs(actual['macro_f1']-result['test']['macro_f1'])>1e-12:raise ValueError('Saved metric does not match predictions')
            pooled[name].append({'seed':seed,'run':run_id,'validation_macro_f1':result['validation']['macro_f1'],
                                 'test_macro_f1':actual['macro_f1'],'accuracy':actual['accuracy'],'fit_seconds':result['fit_seconds'],
                                 'per_style_f1':{style:actual['classification_report'][style]['f1-score'] for style in config['id_to_label'].values()}})
            evidence[str(path.relative_to(PROJECT_ROOT))]=file_hash(path)
            if seed==42:seed42[name]=prediction
    expected={'dummy_prior','word_tfidf_logreg','word_tfidf_linear_svc','char_tfidf_linear_svc',*[s['id'] for s in plan['models']]}
    if set(pooled)!=expected:raise ValueError('Unexpected or missing model family')
    aggregated={}
    for name,rows in pooled.items():
        if sorted(r['seed'] for r in rows)!=sorted(plan['seeds']):raise ValueError('Missing or duplicate seed')
        item={'runs':rows}
        for key in ['validation_macro_f1','test_macro_f1','accuracy','fit_seconds']:
            values=[r[key] for r in rows]
            item[key+'_mean']=float(np.mean(values));item[key+'_sample_std']=float(np.std(values,ddof=1))
        aggregated[name]=item
    chosen=max((n for n in aggregated if n!='dummy_prior'),key=lambda n:aggregated[n]['validation_macro_f1_mean'])
    bootstrap=paired_group_bootstrap(test,seed42,'char_tfidf_linear_svc')
    out.mkdir(parents=True)
    write_json(out/'results.json',aggregated);write_json(out/'bootstrap.json',bootstrap)
    write_json(out/'selection.json',{'family':chosen,'representative_seed':42,
        'criterion':'Highest mean validation Macro-F1 across all three seeds; seed 42 fixed in advance for deployment, never best test seed.'})
    write_json(out/'provenance.json',{'suite':str(suite),'plan_sha256':protocol['plan_sha256'],
              'manifest_sha256':config['manifest_sha256'],'prediction_hashes':evidence})
    tr=[];ci=[]
    for name,result in aggregated.items():
        tr.append('<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{result[k+"_mean"]:.4f} ± {result[k+"_sample_std"]:.4f}</td>' for k in ['validation_macro_f1','test_macro_f1','accuracy'])+'</tr>')
        values=bootstrap['models'][name];interval=values['macro_f1_95_percentile_interval'];delta=values['paired_macro_f1_difference_vs_reference_95_interval']
        ci.append(f'<tr><td>{html.escape(name)}</td><td>{interval[0]:.4f} — {interval[1]:.4f}</td><td>{delta[0]:+.4f} — {delta[1]:+.4f}</td></tr>')
    content='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Повторные эксперименты: три seed</title>
<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;color:#18334c}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cad5de;padding:12px;text-align:left}th{background:#eaf2f8}.note{padding:18px;background:#fff4da}</style>
<h1>Сравнение моделей на общей версии text_only_v4_shared</h1>
<p class="note">Разметка предварительная. Это ранее использованный внутренний test, а не новая независимая оценка. Размер: 610 документов; train 370, validation 120, test 120. Все модели получают одинаковые тексты. У трёх train-фрагментов удалены последние 2–6 слов для совместимости с mBERT; документы и разбиения сохранены.</p>
<h2>Все три заранее выбранных seed: 42, 43, 44</h2><p>Среднее ± выборочное стандартное отклонение. Во всех нейросетевых запусках две эпохи; лучшая эпоха определяется по validation. Отчёт включает каждый запланированный запуск. Малый разброс по seed не устраняет смещение корпуса.</p>
<table><tr><th>Модель</th><th>Validation Macro-F1</th><th>Test Macro-F1</th><th>Test accuracy</th></tr>'''+''.join(tr)+'''</table>
<h2>Неопределённость на test: фиксированный seed 42</h2><p>2000 повторов bootstrap с пересэмплированием групп дубликатов внутри классов. Для парных разностей использованы одни и те же повторы. Столбец разности сравнивает модель с символьным SVM. Интервал, включающий ноль, не позволяет уверенно установить направление различия в этой проверке. Интервалы не учитывают множественные сравнения, смещение источников и разброс обучения.</p>
<table><tr><th>Модель</th><th>95% интервал Macro-F1</th><th>95% интервал разности с char SVM</th></tr>'''+''.join(ci)+f'''</table><p>По среднему validation выбрана модель {html.escape(chosen)}; для демонстрации используется заранее установленный seed 42.</p><p>Manifest: {config['manifest_sha256']}</p><p><a href="results.json">Все запуски</a> · <a href="bootstrap.json">Протокол bootstrap</a> · <a href="selection.json">Правило выбора</a></p></html>'''
    (out/'report.html').write_text(content,encoding='utf-8')
    print(json.dumps({'selected_by_validation':chosen,'models':len(aggregated),'seeds':plan['seeds']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();summarize(a.suite,a.out_dir)
