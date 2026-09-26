"""Post-run diagnostics; never modifies data, split or model selection."""
import argparse
import json
from pathlib import Path
from kazstyle.settings import project_path

from kazstyle.data.corpus import load_manifest,write_json
from kazstyle.evaluation.reports import metrics


def diagnose(dataset,out):
    if out.exists():raise FileExistsError(out)
    frame,config=load_manifest(dataset)
    train=frame[frame.split=='train'];test=frame[frame.split=='test']
    overlaps={}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        left,right=frame[frame.split==a],frame[frame.split==b]
        overlaps[f'{a}/{b}']={key:len(set(left[key])&set(right[key]))
                              for key in ['doc_id','content_hash','excerpt_hash','duplicate_group_id']}
    if any(n for pair in overlaps.values() for n in pair.values()):
        raise ValueError('Unexpected cross-split overlap')
    lookup=train.groupby('source_domain').label.agg(lambda x:int(x.mode().iloc[0])).to_dict()
    fallback=int(train.label.mode().iloc[0])
    predictions=[lookup.get(domain,fallback) for domain in test.source_domain]
    result={'manifest_sha256':config['manifest_sha256'],'overlaps':overlaps,
            'source_only_diagnostic':metrics(test.label,predictions,config),
            'lookup_learned_from_train':lookup,
            'interpretation':'Metadata-only diagnostic, NOT a text classifier. A perfect source lookup demonstrates source-label confounding; it does not prove which features a text model used.',
            'next_action':'Build human-reviewed multi-source labels and a separately fixed external test before generalization claims.'}
    out.mkdir(parents=True);write_json(out/'diagnostics.json',result)
    (out/'report.md').write_text('# Проверка ограничений эксперимента\n\n'
        'Пересечения документов, точных текстов и найденных групп повторов между частями: 0.\n\n'
        f"Диагностическое правило, которое видит **только домен сайта**, даёт test Macro-F1 = {result['source_only_diagnostic']['macro_f1']:.4f}. "
        'Соответствие домена классу построено только по train. Это не текстовая модель и не результат для главной таблицы. '
        'Проверка показывает, что структура корпуса допускает решение по источнику; '
        'какие именно признаки использовал каждый текстовый классификатор, она не устанавливает.\n\n'
        'Нужны ручная проверка меток и разные источники каждого стиля.\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=project_path('data/processed/pilot_v2'))
    p.add_argument('--out-dir',type=Path,default=project_path('reports/pilot_v2_diagnostics'))
    a=p.parse_args();diagnose(a.dataset,a.out_dir)
