"""Inspect saved linear weights and exact additive contributions; never refit."""
import argparse
import html
import json
from pathlib import Path

import joblib
import numpy as np

from kazstyle.data.corpus import load_manifest, file_hash, write_json, make_excerpt
from kazstyle.data.quality import clean_text
from kazstyle.models.tokenization import load_dataset_tokenizer, tokenizer_specs, fit_shared_text


def explain(pipeline, text, top=12):
    vectorizer=pipeline.named_steps['tfidf'];classifier=pipeline.named_steps['classifier']
    vector=vectorizer.transform([text]).tocsr()
    names=vectorizer.get_feature_names_out()
    coefficients=classifier.coef_
    scores=np.asarray(pipeline.decision_function([text]))[0]
    if coefficients.shape[0]!=len(classifier.classes_):
        raise ValueError('This audit requires a multiclass linear classifier')
    result={}
    for index,label in enumerate(classifier.classes_):
        contributions=vector.data*coefficients[index,vector.indices]
        restored=float(classifier.intercept_[index]+contributions.sum())
        if not np.isclose(restored,scores[index],rtol=1e-8,atol=1e-9):
            raise ValueError('Contributions do not reproduce decision_function')
        ordered=np.argsort(contributions)[::-1]
        result[int(label)]={'decision_score':float(scores[index]),'intercept':float(classifier.intercept_[index]),
            'positive_contributions':[{'feature':str(names[vector.indices[i]]),'contribution':float(contributions[i])}
                                      for i in ordered if contributions[i]>0][:top],
            'negative_contributions':[{'feature':str(names[vector.indices[i]]),'contribution':float(contributions[i])}
                                      for i in ordered[::-1] if contributions[i]<0][:top]}
    return {'predicted_label':int(pipeline.predict([text])[0]),'classes':result,
            'nonzero_input_features':vector.nnz,'additive_scores_verified':True}


def audit(dataset,run,examples,out):
    if out.exists():raise FileExistsError(out)
    frame,config=load_manifest(dataset)
    constraints=[(load_dataset_tokenizer(dataset,config,s['model_name'])[0],s['max_tokens']) for s in tokenizer_specs(config)]
    examples_json=json.loads(examples.read_text(encoding='utf-8'))
    examples_rows=[]
    for sample in examples_json['examples']:
        cleaned,_=clean_text(sample['text'])
        excerpt,_=make_excerpt(cleaned,constraints[0][0],config['max_words'],constraints[0][1])
        examples_rows.append({**sample,'analyzed_text':fit_shared_text(excerpt,constraints)})
    models={};names=config['id_to_label']
    for name in ['word_tfidf_logreg','word_tfidf_linear_svc','char_tfidf_linear_svc']:
        bundle=joblib.load(run/(name+'.joblib'))
        if bundle['manifest_sha256']!=config['manifest_sha256']:raise ValueError('Wrong dataset for weights')
        pipe=bundle['pipeline'];vectorizer=pipe.named_steps['tfidf'];classifier=pipe.named_steps['classifier']
        vocabulary=vectorizer.get_feature_names_out()
        frequencies=np.asarray((vectorizer.transform(frame[frame.split=='train'].text)!=0).sum(axis=0)).ravel()
        top_features={}
        for index,label in enumerate(classifier.classes_):
            indices=np.argsort(classifier.coef_[index])[::-1][:20]
            top_features[names[str(label)]]=[{'feature':str(vocabulary[i]),'weight':float(classifier.coef_[index,i]),
                                            'train_documents_with_feature':int(frequencies[i])} for i in indices]
        models[name]={'top_positive_weights':top_features,'examples':{
            row['id']:explain(pipe,row['analyzed_text']) for row in examples_rows}}
    out.mkdir(parents=True)
    write_json(out/'audit.json',{'purpose':'Post-hoc diagnostics on known user examples; not model selection or unseen evaluation',
        'manifest_sha256':config['manifest_sha256'],'examples_sha256':file_hash(examples),
        'model_hashes':{p.name:file_hash(p) for p in run.glob('*.joblib')},'models':models,'examples':examples_rows})
    sections=[]
    for row in examples_rows:
        pieces=[]
        for name,record in models.items():
            detail=record['examples'][row['id']];pred=detail['predicted_label']
            features=detail['classes'][pred]['positive_contributions']
            joined=', '.join(html.escape(repr(f['feature']))+f" ({f['contribution']:+.3f})" for f in features)
            pieces.append(f'<h3>{html.escape(name)} → {html.escape(names[str(pred)])}</h3><p>{joined}</p>')
        sections.append('<h2>'+html.escape(row['id'])+'</h2><p>Предварительно ожидается: '+html.escape(row['expected'])+
                        '</p><blockquote>'+html.escape(row['analyzed_text'])+'</blockquote>'+''.join(pieces))
    for name,record in models.items():
        rows=[]
        for style,features in record['top_positive_weights'].items():
            rows.append('<tr><th>'+html.escape(style)+'</th><td>'+', '.join(html.escape(repr(f['feature']))+
                        f" ({f['weight']:+.3f}; n={f['train_documents_with_feature']})" for f in features)+'</td></tr>')
        sections.append('<h2>Наибольшие положительные веса: '+html.escape(name)+'</h2><table>'+''.join(rows)+'</table>')
    (out/'report.html').write_text('''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Почему линейная модель даёт такой ответ</title>
<style>body{max-width:1050px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;color:#243549}h2{margin-top:42px}blockquote{padding:16px;background:#eef3f7}td,th{padding:12px;border:1px solid #ccd6df;vertical-align:top}table{border-collapse:collapse}th{text-align:left}</style>
<h1>Разбор признаков линейных моделей</h1><p>Сохранённые модели не переобучались. Для каждого известного примера показаны признаки, которые внесли наибольший положительный вклад в оценку выбранного класса. Вклад равен TF-IDF × вес. Проверено, что сумма всех вкладов и свободного члена воспроизводит decision_function. Эти оценки не являются вероятностями. Наличие признака не доказывает, что он является лингвистическим признаком стиля.</p>
<p>Глобальная таблица показывает веса, а не вклады в конкретный текст; n — число обучающих документов с данным признаком. Высокие веса тематических слов или формулировок одного сайта помогают найти смещение корпуса.</p>'''+''.join(sections)+'</html>',encoding='utf-8')
    print(json.dumps({name:{key:names[str(value['predicted_label'])] for key,value in record['examples'].items()} for name,record in models.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--examples',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();audit(a.dataset,a.run,a.examples,a.out_dir)
