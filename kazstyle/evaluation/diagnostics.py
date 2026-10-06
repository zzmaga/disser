"""Known user examples and fixed name/date perturbations, without retraining."""
import argparse
import html
import json
from pathlib import Path

from kazstyle.data.corpus import file_hash, write_json, write_jsonl
from kazstyle.inference.service import InferenceService


def fixed_cases(examples):
    cases=[{**r,'base_id':r['id'],'change':'Исходный пример пользователя'} for r in examples]
    application=next(r for r in examples if r['id']=='user_school_application')
    variants=[('year_2019','Год 2026 → 2019',{'2026':'2019'}),
              ('year_2035','Год 2026 → 2035',{'2026':'2035'}),
              ('pupil_name','Заменено имя ученика',{'Ерлан Амановтан':'Әли Сәрсеновтен'}),
              ('director_name','Заменено имя директора',{'Б. Қалиевке':'А. Әбдіровке'}),
              ('names_and_year','Заменены оба имени и год',{'Ерлан Амановтан':'Әли Сәрсеновтен','Б. Қалиевке':'А. Әбдіровке','2026':'2019'}),
              ('different_day','Заменены дни в заявлении',{'5-қазан':'12-қазан','04.10.':'11.10.'})]
    for suffix,title,replacements in variants:
        text=application['text']
        for old,new in replacements.items():
            if old not in text:raise ValueError('Diagnostic source text changed: '+old)
            text=text.replace(old,new)
        cases.append({'id':application['id']+'_'+suffix,'base_id':application['id'],
                      'expected':application['expected'],'text':text,'change':title})
    return cases


def evaluate(examples_path,out,deployment=None):
    if out.exists():raise FileExistsError(out)
    examples=json.loads(examples_path.read_text(encoding='utf-8'))
    cases=fixed_cases(examples['examples'])
    service=InferenceService(deployment=deployment)
    out.mkdir(parents=True)
    write_json(out/'protocol.json',{'purpose':'Known user examples and synthetic perturbations, not an independent benchmark',
        'label_origin':'User examples; variants inherit the expected label and preserve the application wording',
        'examples_sha256':file_hash(examples_path),'manifest_sha256':service.config['manifest_sha256'],
        'models':service.models,'cases_fixed_before_inference':cases,
        'interpretation':'Changed prediction after a name/date edit is sensitivity, not proof of the exact causal feature. No model tuning follows this check.'})
    records=[]
    for case in cases:
        response=service.classify(case['text'],compare=True)
        records.append({**case,'response':response})
        print(json.dumps({'id':case['id'],'expected':case['expected'],
              'predictions':{r['model']:r['style'] for r in response['results']}},ensure_ascii=True),flush=True)
    write_jsonl(out/'predictions.jsonl',records)
    original=next(r for r in records if r['id']=='user_school_application')
    baseline={r['model']:r['style'] for r in original['response']['results']}
    changes={key:[row['id'] for row in records if row['base_id']=='user_school_application' and row['id']!=row['base_id']
                  and next(r['style'] for r in row['response']['results'] if r['model']==key)!=style]
             for key,style in baseline.items()}
    write_json(out/'sensitivity.json',{'original_application_predictions':baseline,'changed_predictions':changes,
        'variants_per_model':6,'meaning':'Counts concern six fixed variants of one known application, not population error rates.'})
    rows=[]
    for row in records:
        cells=''.join('<td class="'+('ok' if r['style']==row['expected'] else 'error')+'">'+html.escape(r['style_ru'])+'</td>'
                      for r in row['response']['results'])
        rows.append('<tr><td>'+html.escape(row['id'])+'<br>'+html.escape(row['change'])+'</td><td>'+html.escape(row['expected'])+'</td>'+cells+'</tr>')
    columns=''.join('<th>'+html.escape(spec['name'])+'</th>' for spec in service.models.values())
    details=''.join('<details><summary>'+html.escape(row['id'])+'</summary><p>'+html.escape(row['response']['excerpt'])+'</p></details>' for row in records)
    content='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка примеров пользователя</title>
<style>body{font:16px/1.5 system-ui;margin:36px;color:#203348}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #cad5df;text-align:left;vertical-align:top}th{background:#edf2f7}.ok{background:#eaf6ea}.error{background:#ffe4e4}details{margin:14px 0}</style>
<h1>Известные ошибки и чувствительность к имени и дате</h1><p>Проверяются два ранее сообщённых примера и шесть заранее заданных вариантов одного заявления. Содержание просьбы сохранено; меняются только имена или даты. Ожидаемые стили заданы пользователем. Это диагностика уже известных примеров, а не независимый test. По восьми строкам не рассчитывается итоговая точность диссертационной системы.</p>'''
    (out/'report.html').write_text(content+'<table><tr><th>Пример и изменение</th><th>Ожидается</th>'+columns+'</tr>'+''.join(rows)+'</table><h2>Фактически прочитанные тексты</h2>'+details+'</html>',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--examples',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--deployment',type=Path)
    a=p.parse_args();evaluate(a.examples,a.out_dir,a.deployment)
