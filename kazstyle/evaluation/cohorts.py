"""Evaluate saved test predictions without refitting or changing model selection."""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from kazstyle.data.corpus import load_manifest, write_json
from kazstyle.evaluation.reports import metrics


def evaluate(dataset, runs, out):
    if out.exists():
        raise FileExistsError(out)
    frame, config = load_manifest(dataset)
    train = frame[frame.split == 'train']
    test = frame[frame.split == 'test'].copy()
    test['length_band'] = pd.cut(test.excerpt_words, bins=[0, 39, 79, float('inf')],
                                 labels=['8-39 words', '40-79 words', '80+ words'])
    # Deliberately ignores words: measures the dataset's length/class correlation.
    length_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    length_model.fit(train[['excerpt_words']], train.label)
    result = {'manifest_sha256': config['manifest_sha256'],
              'length_only_diagnostic': metrics(test.label, length_model.predict(test[['excerpt_words']]), config),
              'interpretation': 'Internal-test diagnostics, not external validation. Cohort Macro-F1 uses only classes present in each cohort.',
              'models': {}}
    from sklearn.metrics import f1_score, accuracy_score
    for run in runs:
        provenance = json.loads((run / 'provenance.json').read_text(encoding='utf-8'))
        if provenance['manifest_sha256'] != config['manifest_sha256']:
            raise ValueError('Different dataset in run')
        run_results = json.loads((run / 'results.json').read_text(encoding='utf-8'))
        for name in run_results:
            path = run / (f'{name}_test_predictions.jsonl' if len(run_results) > 1 else 'test_predictions.jsonl')
            predictions = pd.read_json(path, lines=True)
            if set(predictions.sample_id) != set(test.sample_id) or predictions.sample_id.duplicated().any():
                raise ValueError('Missing, extra or duplicate predictions')
            joined = test.merge(predictions[['sample_id', 'y_true', 'y_pred']], on='sample_id', validate='one_to_one')
            if not (joined.label == joined.y_true).all():
                raise ValueError('Prediction ground truth differs from manifest')
            cohorts = {}
            for key in ['length_band', 'source_domain']:
                cohorts[key] = {}
                for value, part in joined.groupby(key, observed=True):
                    labels = sorted(part.label.unique().tolist())
                    cohorts[key][str(value)] = {
                        'n': len(part), 'present_labels': labels,
                        'accuracy': float(accuracy_score(part.label, part.y_pred)),
                        'macro_f1_present_classes': float(f1_score(part.label, part.y_pred, labels=labels,
                                                                  average='macro', zero_division=0))}
            result['models'][name] = cohorts
    out.mkdir(parents=True)
    write_json(out / 'cohorts.json', result)
    print(json.dumps({'length_only_macro_f1': result['length_only_diagnostic']['macro_f1'],
                      'models': list(result['models'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.dataset, args.runs, args.out_dir)
