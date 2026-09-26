"""Train fixed classical baselines on ONE versioned two/three-class manifest."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from kazstyle.settings import project_path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from kazstyle.data.corpus import load_manifest, write_json
from kazstyle.evaluation.reports import metrics, provenance, render_report, save_predictions


def model_specs(seed=42):
    # Fixed before test evaluation. No grid search, test-driven tuning or scores in code.
    def word():
        return TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=80000)
    return {
        'dummy_prior':Pipeline([('tfidf',word()),('classifier',DummyClassifier(strategy='prior'))]),
        'word_tfidf_logreg':Pipeline([('tfidf',word()),('classifier',LogisticRegression(C=1,max_iter=2000,random_state=seed))]),
        'word_tfidf_linear_svc':Pipeline([('tfidf',word()),('classifier',LinearSVC(C=1,random_state=seed))]),
        'char_tfidf_linear_svc':Pipeline([('tfidf',TfidfVectorizer(analyzer='char',ngram_range=(3,5),
                                         min_df=2,max_features=100000,sublinear_tf=True)),
                                        ('classifier',LinearSVC(C=1,random_state=seed))]),
    }


def train(dataset, out, seed=42):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite run: {out}')
    frame,config = load_manifest(dataset)
    train_set = frame[frame.split=='train']
    validation = frame[frame.split=='validation']
    test = frame[frame.split=='test']
    out.mkdir(parents=True)
    specs = model_specs(seed)
    params = {name:{key:str(value) for key,value in pipe.get_params(deep=True).items()}
              for name,pipe in specs.items()}
    write_json(out/'protocol.json',{'models':params,'selection':'all configurations fixed before test',
                                   'seed':seed,'dataset_config':config})
    write_json(out/'provenance.json',provenance(dataset,{'seed':seed}))
    results = {}
    for name,pipe in specs.items():
        start = time.perf_counter()
        pipe.fit(train_set.text,train_set.label)
        fit_seconds = time.perf_counter()-start
        val_pred = pipe.predict(validation.text)
        # There is no parameter adjustment based on validation or test in this run.
        start = time.perf_counter()
        test_pred = pipe.predict(test.text)
        predict_seconds = time.perf_counter()-start
        bundle = {'pipeline':pipe,'dataset_config':config,'model_name':name,
                  'input_policy':config['excerpt_policy'],'manifest_sha256':config['manifest_sha256']}
        joblib.dump(bundle,out/f'{name}.joblib')
        restored = joblib.load(out/f'{name}.joblib')
        if not np.array_equal(test_pred[:10],restored['pipeline'].predict(test.text.iloc[:10])):
            raise RuntimeError('Serialization changed predictions')
        result = {'validation':metrics(validation.label,val_pred,config),
                  'test':metrics(test.label,test_pred,config),'fit_seconds':fit_seconds,
                  'test_predict_seconds':predict_seconds,
                  'test_ms_per_document':1000*predict_seconds/len(test),'reload_verified':True}
        results[name] = result
        save_predictions(out/f'{name}_validation_predictions.jsonl',validation,val_pred,config)
        save_predictions(out/f'{name}_test_predictions.jsonl',test,test_pred,config)
        print(f"{name}: validation={result['validation']['macro_f1']:.4f} test={result['test']['macro_f1']:.4f} fit={fit_seconds:.2f}s",flush=True)
        write_json(out/'results.json',results)
    # Deployment choice uses validation only, even though all fixed test results are reported.
    best = max((n for n in results if n!='dummy_prior'),key=lambda n:results[n]['validation']['macro_f1'])
    write_json(out/'selection.json',{'model':best,'criterion':'highest validation Macro-F1; insertion-order tie break',
                                    'model_file':f'{best}.joblib'})
    render_report(out,frame,config,results)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=project_path('data/processed/pilot_v2'))
    p.add_argument('--out-dir',type=Path,default=project_path('artifacts/pilot_v2_classical'))
    p.add_argument('--seed',type=int,default=42)
    a = p.parse_args()
    train(a.dataset,a.out_dir,a.seed)


if __name__=='__main__':
    main()
