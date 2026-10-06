"""Compose a new development pool from hash-bound sources and explicit exclusions."""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
from kazstyle.data.build_v3 import diverse_sample
from kazstyle.data.corpus import digest, file_hash, write_json, write_jsonl


def merge(recipe_path, out):
    if out.exists():
        raise FileExistsError(out)
    recipe = json.loads(recipe_path.read_text(encoding='utf-8'))
    if recipe.get('purpose') != 'provisional_development_pool':
        raise ValueError('This command only builds a provisional development pool')
    by_id, repeated = {}, []
    for item in recipe['inputs']:
        path = Path(item['path'])
        if file_hash(path) != item['sha256']:
            raise ValueError('Recipe source checksum mismatch')
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if str(row.get('usage_status','')).startswith('reserved'):
                raise ValueError('Reserved evaluation/source material cannot enter a development pool')
            key = row['doc_id']
            if key in by_id:
                first = by_id[key]
                if (first['style'], first['text']) != (row['style'], row['text']):
                    raise ValueError(f'Conflicting document versions: {key}')
                repeated.append({'doc_id': key, 'duplicate_input': str(path)})
            else:
                by_id[key] = {**row, 'pool_input_file': str(path), 'pool_input_text_sha256': digest(row['text'])}
    excluded = {r['doc_id']: r for r in recipe.get('exclusions', [])}
    if len(excluded) != len(recipe.get('exclusions', [])) or not set(excluded).issubset(by_id):
        raise ValueError('Duplicate or unknown exclusion IDs')
    rejected = []
    for doc_id, decision in excluded.items():
        row = by_id.pop(doc_id)
        if decision.get('text_sha256') != digest(row['text']) or not decision.get('reason'):
            raise ValueError('Exclusion needs a matching text hash and reason')
        rejected.append({**row, 'exclusion': decision})
    rows = list(by_id.values())
    before = Counter(r['style'] for r in rows)
    cap = recipe['max_per_style']
    if type(cap) is not int or cap < 40:
        raise ValueError('Invalid candidate cap')
    frame = pd.DataFrame(rows)
    selected = pd.concat([diverse_sample(part, min(cap, len(part)), recipe['seed'])
                          for _, part in frame.groupby('style')], ignore_index=True)
    # A JSON null is distinct from a shared missing-author value; forbid NaN output.
    chosen = selected.astype(object).where(pd.notna(selected), None).to_dict('records')
    out.mkdir(parents=True)
    shutil.copy2(recipe_path,out/'recipe.json');shutil.copy2(__file__,out/'code_snapshot.py')
    write_jsonl(out/'candidates.jsonl', chosen);write_jsonl(out/'excluded.jsonl', rejected)
    write_json(out/'duplicate_inputs.json', repeated)
    summary = {'documents_before_cap': len(rows), 'by_style_before_cap': dict(before),
               'selected_candidates': len(chosen), 'by_style': dict(Counter(r['style'] for r in chosen)),
               'explicit_exclusions': len(rejected), 'repeated_input_records': len(repeated),
               'max_per_style': cap, 'seed': recipe['seed'],
               'selection': 'round_robin_source_and_genre; deterministic within strata',
               'recipe_sha256': file_hash(recipe_path), 'candidates_sha256': file_hash(out/'candidates.jsonl'),
               'expert_verified': False, 'new_final_test': False, 'models_used_for_selection': False,
               'limitations': recipe['limitations']}
    write_json(out/'summary.json', summary)
    write_json(out/'COMPLETE.json', {'candidates_sha256': summary['candidates_sha256']})
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args();merge(args.recipe,args.out_dir)
