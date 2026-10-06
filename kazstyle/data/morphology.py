"""Annotate the fixed Kazakh inputs locally with pinned Stanza morphology models."""
import argparse
import json
import time
from pathlib import Path

from kazstyle.data.corpus import digest, file_hash, load_manifest, write_json, write_jsonl
from kazstyle.settings import PROJECT_ROOT


def project_words(words, field):
    if field not in {'text', 'lemma'}:
        raise ValueError('Unsupported projection')
    if not words or any(not isinstance(w.get(field), str) or not w[field].strip() for w in words):
        raise ValueError('Missing morphological output; silent identity fallback is forbidden')
    return ' '.join(w[field] for w in words)


def aligned_annotations(frame, records):
    indexed = {row['sample_id']: row for row in records}
    if len(indexed) != len(records) or set(indexed) != set(frame.sample_id):
        raise ValueError('Annotations must match every frozen sample exactly once')
    ordered = []
    for row in frame.itertuples():
        annotation = indexed[row.sample_id]
        if annotation['text_sha256'] != digest(row.text) or annotation['doc_id'] != row.doc_id:
            raise ValueError('Annotation belongs to different text/document')
        for field in ['text', 'lemma']:
            project_words(annotation['words'], field)
        ordered.append(annotation)
    return ordered


def annotate(plan_path, out):
    import stanza
    import torch
    from kazstyle.evaluation.reports import save_provenance
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if stanza.__version__ != plan['stanza_version']:
        raise ValueError('Wrong Stanza version')
    dataset = PROJECT_ROOT/plan['dataset']
    frame, config = load_manifest(dataset)
    if config['manifest_sha256'] != plan['manifest_sha256']:
        raise ValueError('Wrong dataset')
    model_dir = PROJECT_ROOT/plan['model_dir']
    weights = sorted(model_dir.rglob('*.pt'))
    if not weights:
        raise ValueError('Download the declared Stanza models first')
    hashes = {str(path.relative_to(model_dir)): file_hash(path) for path in [model_dir/'resources.json', *weights]}
    torch.set_num_threads(2)
    torch.manual_seed(plan['seed'])
    pipeline = stanza.Pipeline('kk', dir=str(model_dir), package=None,
        processors=plan['processors'], resources_version=plan['resources_version'],
        download_method=None, use_gpu=False, verbose=False)
    out.mkdir(parents=True)
    save_provenance(out, dataset, {'plan_sha256': file_hash(plan_path), 'stanza': stanza.__version__,
        'processors': plan['processors'], 'model_hashes': hashes, 'preprocessing_device': 'cpu'})
    write_json(out/'protocol.json', {'plan': plan, 'plan_sha256': file_hash(plan_path),
        'model_files_sha256': hashes, 'scope': 'Pretrained annotation only; no fitting on style labels'})
    rows = []
    started = time.perf_counter()
    for i, row in enumerate(frame.itertuples(), 1):
        doc = pipeline(row.text)
        words = [{'text': w.text, 'lemma': w.lemma, 'upos': w.upos, 'xpos': w.xpos,
                  'feats': w.feats, 'sentence': j}
                 for j, sentence in enumerate(doc.sentences) for w in sentence.words]
        project_words(words, 'lemma')
        rows.append({'sample_id': row.sample_id, 'doc_id': row.doc_id,
                     'text_sha256': digest(row.text), 'words': words})
        if i % 50 == 0:
            print(f'[morphology] {i}/{len(frame)} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    aligned_annotations(frame, rows)
    write_jsonl(out/'annotations.jsonl', rows)
    word_count = sum(len(r['words']) for r in rows)
    changes = sum(w['text'].casefold() != w['lemma'].casefold() for r in rows for w in r['words'])
    summary = {'documents': len(rows), 'words': word_count, 'changed_lemmas_ignoring_case': changes,
        'changed_fraction': changes/word_count, 'annotation_seconds': time.perf_counter()-started,
        'annotations_sha256': file_hash(out/'annotations.jsonl'), 'manifest_sha256': config['manifest_sha256'],
        'limitations': 'Predicted lemmas and morphology, not human gold. No evaluation of annotation accuracy on this corpus.'}
    write_json(out/'summary.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    annotate(a.plan, a.out_dir)
