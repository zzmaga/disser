"""Prepare a verifiable deployment profile from a completed repeated experiment."""
import argparse
import json
from pathlib import Path

from kazstyle.data.corpus import file_hash, load_manifest, write_json
from kazstyle.settings import PROJECT_ROOT, artifact_path


FAMILIES={
    'kaz_roberta_last':('roberta','Kaz-RoBERTa','transformer'),
    'kaz_roberta_concat4':('roberta_concat4','Kaz-RoBERTa · 4 слоя','transformer'),
    'mbert_last':('mbert','mBERT','transformer'),
    'word_tfidf_logreg':('logreg','TF-IDF + Logistic Regression','classical'),
    'word_tfidf_linear_svc':('word_svm','TF-IDF слов + SVM','classical'),
    'char_tfidf_linear_svc':('char_svm','TF-IDF символов + SVM','classical'),
}


def prepare(suite,summary,out):
    if out.exists():raise FileExistsError(out)
    status=json.loads((suite/'status.json').read_text(encoding='utf-8'))
    protocol=json.loads((suite/'protocol.json').read_text(encoding='utf-8'))
    evidence=json.loads((summary/'provenance.json').read_text(encoding='utf-8'))
    selection=json.loads((summary/'selection.json').read_text(encoding='utf-8'))
    results=json.loads((summary/'results.json').read_text(encoding='utf-8'))
    if status['status']!='complete' or evidence['plan_sha256']!=protocol['plan_sha256']:
        raise ValueError('Only the complete matching experiment can be deployed')
    dataset=PROJECT_ROOT/protocol['plan']['dataset'];_,config=load_manifest(dataset)
    if evidence['manifest_sha256']!=config['manifest_sha256']:raise ValueError('Dataset changed')
    seed=selection['representative_seed']
    if seed!=42:raise ValueError('The representative seed was fixed at 42 before training')
    chosen=max((name for name in results if name in FAMILIES),key=lambda name:results[name]['validation_macro_f1_mean'])
    if selection['family']!=chosen:raise ValueError('Selection differs from mean validation rule')
    models={}
    for family,(key,name,kind) in FAMILIES.items():
        matches=[r for r in results[family]['runs'] if r['seed']==seed]
        if len(matches)!=1:raise ValueError('Expected exactly one representative run')
        run_id=matches[0]['run'];run=artifact_path(run_id)
        marker=json.loads((run/'COMPLETE.json').read_text(encoding='utf-8'))
        if marker['plan_sha256']!=protocol['plan_sha256'] or marker['manifest_sha256']!=config['manifest_sha256']:
            raise ValueError('Run not completed under this protocol')
        if run_id not in status['completed']:raise ValueError('Unknown completed run')
        filename=family+'.joblib' if kind=='classical' else 'best_model.pt'
        if kind=='transformer':
            verified=json.loads((run/'reload_verification.json').read_text(encoding='utf-8'))
            if verified.get('offline_reload_verified') is not True or verified['manifest_sha256']!=config['manifest_sha256']:
                raise ValueError('Offline restoration has not been verified')
            if verified.get('checked_labels')!=list(range(len(config['styles']))):
                raise ValueError('Offline check must include every style')
        spec={'name':name,'kind':kind,'run':run_id,'weights_sha256':file_hash(run/filename)}
        if kind=='classical':spec['file']=filename
        models[key]=spec
    profile={'dataset':protocol['plan']['dataset'],'default_model':FAMILIES[chosen][0],
             'models':models,'selection':selection,'experiment_plan_sha256':protocol['plan_sha256'],
             'summary_sha256':file_hash(summary/'results.json'),
             'evidence':{'suite':suite.resolve().relative_to(PROJECT_ROOT).as_posix(),
                         'summary':summary.resolve().relative_to(PROJECT_ROOT).as_posix()}}
    out.parent.mkdir(parents=True,exist_ok=True);write_json(out,profile)
    print(json.dumps({'profile':str(out),'default':profile['default_model'],'models':list(models)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True);p.add_argument('--out-file',type=Path,required=True)
    a=p.parse_args();prepare(a.suite,a.summary,a.out_file)
