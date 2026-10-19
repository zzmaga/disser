"""Build an English working manuscript from saved evidence, without training.

The prose is a separate template. Numbers come from checksummed experiments and
the registered corpus; a working draft is never labelled submission-ready.
"""
import argparse
from datetime import datetime,timezone
import html
import json
import os
from pathlib import Path
from markupsafe import Markup

from kazstyle.data.corpus import file_hash,write_json
from kazstyle.settings import PROJECT_ROOT,evidence_path
from kazstyle.experiment_history import history_index
from kazstyle.evaluation.rendering import render_template, table


def build(out):
    root=PROJECT_ROOT;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    registry=json.loads((root/'configs/corpus_registry.json').read_text(encoding='utf-8'))
    corpus=root/registry['active_document_corpus']
    manifest=json.loads((corpus/'COMPLETE.json').read_text(encoding='utf-8'))
    if file_hash(corpus/'summary.json')!=manifest['summary.json']:
        raise ValueError('Corpus summary differs from its immutable manifest')
    summary=json.loads((corpus/'summary.json').read_text(encoding='utf-8'))
    result_dir=evidence_path('reports/expanded_v6/repeated')
    results=json.loads((result_dir/'results.json').read_text(encoding='utf-8'))
    deployment=json.loads((root/'configs/deployment.json').read_text(encoding='utf-8'))
    if file_hash(evidence_path('reports/text_only_v4_repeated/results.json'))!=deployment['summary_sha256']:
        raise ValueError('Active deployment results changed')
    provenance=json.loads((result_dir/'provenance.json').read_text(encoding='utf-8'))
    suite=json.loads((evidence_path(provenance['suite'])/'protocol.json').read_text(encoding='utf-8'))
    if suite['plan_sha256']!=provenance['plan_sha256']:
        raise ValueError('Development summary belongs to a different suite')
    if file_hash(root/suite['plan']['dataset']/'manifest.jsonl')!=provenance['manifest_sha256']:
        raise ValueError('Development experiment manifest changed')
    for path,expected in provenance['prediction_hashes'].items():
        if file_hash(evidence_path(path))!=expected:
            raise ValueError('Saved development predictions changed: '+path)
    heldout=evidence_path('reports/expanded_v6/source_holdout/results.json')
    held=json.loads(heldout.read_text(encoding='utf-8'))
    labels={'official':'Official and business','scientific':'Scientific','publicistic':'Publicistic',
            'literary':'Literary','colloquial':'Colloquial'}
    names={'dummy_prior':'Majority/prior baseline',
           'word_tfidf_logreg':'Word TF-IDF + Logistic Regression',
           'word_tfidf_linear_svc':'Word TF-IDF + Linear SVM',
           'char_tfidf_linear_svc':'Character TF-IDF + Linear SVM',
           'mbert_last':'mBERT','kaz_roberta_last':'Kaz-RoBERTa last',
           'kaz_roberta_concat4':'Kaz-RoBERTa concat4'}
    corpus_table=table(['Functional style','Provisionally accepted documents'],
                       [(labels[s],f'{n:,}') for s,n in summary['by_style'].items()])
    results_table=table(['Classifier','Validation Macro-F1, mean ± SD','Development test Macro-F1, mean ± SD'],[
        [names[key],f"{r['validation_macro_f1_mean']:.4f} ± {r['validation_macro_f1_sample_std']:.4f}",
         f"{r['test_macro_f1_mean']:.4f} ± {r['test_macro_f1_sample_std']:.4f}"] for key,r in results.items()])
    held_table=table(['Classifier trained for the source-holdout diagnostic','Validation Macro-F1','Held-out-source Macro-F1'],[
        [names.get(key,'Majority/prior baseline'),f"{r['validation_macro_f1']:.4f}",f"{r['test']['macro_f1']:.4f}"]
        for key,r in held.items()])
    def link(path,label):
        parts=Path(path).parts
        if len(parts)>1 and parts[0]=='reports' and parts[1] in history_index(root).get('reports',[]):
            path='docs/EXPERIMENTS.md'
        path=evidence_path(path)
        if not path.exists():raise FileNotFoundError(path)
        href=Path(os.path.relpath(path,out)).as_posix()
        return '<a href="'+html.escape(href,quote=True)+'">'+html.escape(label)+'</a>'
    evidence_links='<ul>'+''.join('<li>'+link(path,label)+'</li>' for path,label in [
        ('reports/expanded_v6/repeated/report.html','Repeated development experiments and uncertainty'),
        ('reports/expanded_v6/source_holdout/report.html','Source-holdout diagnostic'),
        ('reports/expanded_v6/user_diagnostics/report.html','Known regression cases'),
        ('reports/morphology_ablation_v1/report.html','Morphological preprocessing ablation'),
        ('reports/technical/dissertation_completion_20261018/review_summary.json','Assistant review and source-restoration record'),
        ('reports/technical/dissertation_completion_20261018/source_addendum/manifest.json','Historical repair-source snapshot addendum for the earlier corpus revision'),
        ('reports/technical/dissertation_completion_20261018/curation_followup/report.html','Full-text assistant curation: 888 decisions and reasons'),
        ('configs/evaluation_reservations.json','Registered prospective evaluation embargo'),
        ('data/annotation/reserved_v7_20261018c/summary.json','Reserved candidate groups; reference labels and views still require review'),
        ('configs/reviews/reserved_v7_pilot_20261019.json','AI-assisted review of 50 exact reserved views; not independent human annotation'),
        ('reports/technical/dissertation_completion_20261018/curation_followup/reserved_legacy_overlap/report.html','Exact and shared-passage audit against recorded v4/v6 inputs'),
        ('reports/technical/dissertation_completion_20261018/curation_followup/reserved_development_overlap_closed/report.html','Closed audit between the final development views and reserved candidates'),
        ('reports/technical/dissertation_completion_20261018/encoder_compatibility/mmbert_gpu.json','Discarded train-only mmBERT-small GPU compatibility probe'),
    ])+'</ul>'
    corpus_id=corpus.name
    reservation_dir=root/registry['evaluation_reservation']
    reservation=json.loads((reservation_dir/'summary.json').read_text(encoding='utf-8'))
    frozen=json.loads((reservation_dir/'COMPLETE.json').read_text(encoding='utf-8'))
    if file_hash(reservation_dir/'summary.json')!=frozen['summary.json']:
        raise ValueError('Reservation summary changed')
    if registry.get('reservation_checks'):
        combined=json.loads((root/registry['reservation_checks']).read_text(encoding='utf-8'))
        for path,sha in combined['ledgers'].items():
            if file_hash(root/path)!=sha:raise ValueError('Combined reservation check is stale')
        reservation={**reservation,**combined}
    reservation_table=table(['Functional style','Documents excluded from new training/validation','Remaining development bank'],
        [(labels[s],f'{reservation["embargo_by_style"][s]:,}',f'{reservation["eligible_by_style"][s]:,}')
         for s in labels])
    replacements={
        'CORPUS_ID':corpus_id,'CORPUS_N':f"{sum(summary['by_style'].values()):,}",
        'CORPUS_TABLE':corpus_table,'RESULTS_TABLE':results_table,'HOLDOUT_TABLE':held_table,
        'RESERVATION_TABLE':reservation_table,
        'EMBARGO_N':f'{reservation["embargo_documents"]:,}',
        'EVIDENCE_LINKS':Markup(evidence_links),
    }
    template=render_template('thesis_english.html', **replacements)
    (out/'manuscript.html').write_text(template,encoding='utf-8')
    write_json(out/'evidence.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'status':'working_english_draft_not_ready_for_submission','corpus':registry['active_document_corpus'],
        'corpus_complete_sha256':file_hash(corpus/'COMPLETE.json'),
        'development_results_sha256':file_hash(result_dir/'results.json'),
        'source_holdout_results_sha256':file_hash(heldout),'human_annotations_available':False,
        'reservation_complete_sha256':file_hash(reservation_dir/'COMPLETE.json'),
        'combined_reservation_checks_sha256':file_hash(root/registry['reservation_checks']),
        'template_sha256':file_hash(Path(__file__).with_name('thesis_english.html'))})
    return out/'manuscript.html'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=PROJECT_ROOT/'reports/dissertation_english')
    print(build(parser.parse_args().out_dir))
