"""Development-only comparison of central vs whole-document window decisions."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import time

from kazstyle.data.checkpoint import atomic_json
from kazstyle.data.corpus import digest, file_hash, make_excerpt, write_json, write_jsonl


def aggregate_votes(votes, class_order, central):
    """Word-weighted hard votes; these fractions are not model probabilities."""
    totals = Counter()
    for vote in votes:
        if vote['style'] not in class_order or type(vote['words']) is not int or vote['words'] <= 0:
            raise ValueError('Invalid window vote')
        totals[vote['style']] += vote['words']
    if not totals:
        return {'style':None, 'classified_words':0, 'dominant_vote_word_fraction':None, 'tied':False}
    top = max(totals.values())
    leaders = [style for style in class_order if totals[style] == top]
    chosen = central if central in leaders else leaders[0]
    return {'style':chosen, 'classified_words':sum(totals.values()), 'word_votes':dict(totals),
            'dominant_vote_word_fraction':top/sum(totals.values()), 'tied':len(leaders)>1}


def run(plan_path, out, resume=False):
    from kazstyle.data.export_corpus import load_documents
    from kazstyle.data.quality import clean_text
    from kazstyle.inference.service import InferenceService
    from kazstyle.models.tokenization import fit_shared_text

    plan_path, out = Path(plan_path), Path(out)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan['purpose'] != 'development_window_probe_NOT_final_test' or plan['aggregation'] != 'word_weighted_hard_vote':
        raise ValueError('Unexpected probe protocol')
    profile_path, windows_dir = Path(plan['deployment']), Path(plan['window_plan'])
    for path, expected in [(profile_path,plan['deployment_sha256']),
                           (windows_dir/'COMPLETE.json',plan['window_plan_complete_sha256'])]:
        if file_hash(path) != expected:
            raise ValueError('Frozen probe input changed')
    for name, expected in json.loads((windows_dir/'COMPLETE.json').read_text(encoding='utf-8')).items():
        path = (windows_dir/name).resolve()
        if not path.is_relative_to(windows_dir.resolve()) or file_hash(path) != expected:
            raise ValueError('Window plan integrity mismatch')
    if (out/'COMPLETE.json').exists() or (out.exists() and not resume):
        raise FileExistsError(out)
    descriptor = {'plan_sha256':file_hash(plan_path), 'code_sha256':file_hash(__file__),
                  'inference_code_sha256':file_hash(Path('kazstyle/inference/service.py'))}
    if resume:
        if json.loads((out/'run.json').read_text(encoding='utf-8')) != descriptor:
            raise ValueError('Cannot resume with changed plan or inference code')
    else:
        (out/'documents').mkdir(parents=True)
        write_json(out/'run.json', descriptor)
        shutil.copy2(plan_path, out/'plan.json')
        shutil.copy2(__file__, out/'code_snapshot.py')
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    profile['models'] = {key:profile['models'][key] for key in plan['models']}
    profile['default_model'] = plan['models'][0]
    service = InferenceService(deployment=profile)
    class_order = list(service.config['id_to_label'].values())
    windows = defaultdict(list)
    for line in (windows_dir/'windows.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line);windows[row['doc_id']].append(row)
    corpus = Path(plan['corpus'])
    originals = {row['doc_id']:row for row in load_documents(corpus)}
    tick = time.monotonic()
    documents = []
    for index, (doc_id, parts) in enumerate(windows.items(), 1):
        path = out/'documents'/f'{doc_id}.json'
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
            checksum = saved.pop('record_sha256', None)
            if digest(json.dumps(saved,ensure_ascii=False,sort_keys=True)) != checksum:
                raise ValueError('Saved document result checksum mismatch')
            if saved['plan_sha256'] != descriptor['plan_sha256'] or saved['doc_id'] != doc_id:
                raise ValueError('Unexpected saved document result')
            documents.append(saved)
            continue
        original = originals[doc_id]
        if any(part['parent_text_sha256'] != digest(original['text']) for part in parts):
            raise ValueError('Window parent changed')
        cleaned, _ = clean_text(original['text'])
        central_text, _ = make_excerpt(cleaned, service.tokenizer, service.config['max_words'], service.config['max_tokens'])
        central_text = fit_shared_text(central_text, service.constraints)
        central = service.classify(central_text, compare=True)
        if central['excerpt'] != central_text:
            raise ValueError('Central input unexpectedly transformed')
        central_predictions = {r['model']:r['style'] for r in central['results']}
        predictions, failures = [], []
        for part in parts:
            try:
                prediction = service.classify(part['text'],compare=True)
            except ValueError as exc:
                failures.append({'window_id':part['window_id'],'words':part['words'],'reason':str(exc)})
                continue
            if prediction['excerpt'] != part['text']:
                raise ValueError('Window text changed before inference')
            predictions.append({'window_id':part['window_id'], 'words':part['words'],
                                'text_sha256':digest(part['text']), 'results':prediction['results']})
        aggregate = {}
        for key in plan['models']:
            votes = [{'words':r['words'],'style':next(p['style'] for p in r['results'] if p['model']==key)} for r in predictions]
            aggregate[key] = aggregate_votes(votes,class_order,central_predictions[key])
        saved = {'doc_id':doc_id, 'plan_sha256':descriptor['plan_sha256'], 'style':original['style'],
            'genre':original.get('genre'), 'document_words':len(cleaned.split()),
            'central_words':len(central_text.split()), 'central_text_sha256':digest(central_text),
            'central_predictions':central_predictions, 'window_predictions':predictions,
            'window_failures':failures, 'aggregate':aggregate,
            'window_count':len(parts), 'source_label_status':'provisional_not_gold'}
        atomic_json(path,{**saved,'record_sha256':digest(json.dumps(saved,ensure_ascii=False,sort_keys=True))})
        documents.append(saved)
        if index % 10 == 0:
            print(f'Window probe {index}/{len(windows)} documents, {time.monotonic()-tick:.1f}s this run',flush=True)
    comparison = {}
    for key in plan['models']:
        comparison[key] = {'documents':len(documents),
            'changed_predictions':sum(r['central_predictions'][key]!=r['aggregate'][key]['style'] for r in documents),
            'mixed_window_decisions':sum(len(r['aggregate'][key].get('word_votes',{}))>1 for r in documents),
            'ties':sum(r['aggregate'][key]['tied'] for r in documents)}
    summary = {'documents':len(documents), 'models':comparison,
        'window_failures':sum(len(r['window_failures']) for r in documents),
        'elapsed_seconds_this_run':round(time.monotonic()-tick,3),
        'new_training':False,'deployment_changed':False,'independent_quality_estimate':False,
        'limitations':['Length/genre selected development corpus; may overlap previous training and diagnostic data.',
            'No human window labels. Changed predictions do not establish improvement.',
            'Hard-vote word fractions are not confidence probabilities.',
            'The current website keeps the original central-excerpt policy.']}
    if file_hash(profile_path)!=plan['deployment_sha256'] or file_hash(plan_path)!=descriptor['plan_sha256']:
        raise ValueError('Probe inputs changed during inference')
    write_json(out/'summary.json',summary)
    write_jsonl(out/'documents.jsonl',[{k:v for k,v in r.items() if k!='window_predictions'} for r in documents])
    write_json(out/'COMPLETE.json',{p.relative_to(out).as_posix():file_hash(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();run(args.plan,args.out_dir,args.resume)
