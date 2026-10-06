"""Build an exploratory publisher-holdout diagnostic from frozen development texts.

This does not create an independent final test: source labels are provisional,
and the parent development pool has already been inspected. It measures source
and genre shift together, with new models trained only on this derived train.
"""
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
from kazstyle.data.corpus import load_manifest,validate_manifest,file_hash,write_json,write_jsonl


def take_groups(frame, count, seed):
    """Take up to count rows, consuming entire groups even if their tail is unused."""
    groups=[g.sample(frac=1,random_state=seed) for _,g in frame.groupby('split_group_id',sort=True)]
    # Small groups first reduces unused records; hash order within sizes is fixed.
    groups.sort(key=lambda g:(len(g),str(g.split_group_id.iloc[0])))
    chosen=[];used=set();remaining=count
    for group in groups:
        if remaining<=0:break
        chosen.append(group.head(remaining));used.add(group.split_group_id.iloc[0])
        remaining-=min(remaining,len(group))
    if remaining:raise ValueError(f'Insufficient group-disjoint records: short by {remaining}')
    return pd.concat(chosen), frame[~frame.split_group_id.isin(used)]


def allocate(frame, recipe):
    holdout_domains={d for domains in recipe['heldout_sources'].values() for d in domains}
    is_heldout=frame.source_domain.isin(holdout_domains)
    forbidden=set(frame.loc[is_heldout,'split_group_id'])
    development=frame[~is_heldout & ~frame.split_group_id.isin(forbidden)]
    parts=[]
    for style,domains in recipe['heldout_sources'].items():
        test_pool=frame[(frame['style']==style)&frame.source_domain.isin(domains)]
        test,_=take_groups(test_pool,recipe['test_per_style'],recipe['seed'])
        pool=development[development['style']==style]
        validation,rest=take_groups(pool,recipe['validation_per_style'],recipe['seed'])
        train,_=take_groups(rest,recipe['train_per_style'],recipe['seed'])
        parts.extend([test.assign(split='test'),validation.assign(split='validation'),train.assign(split='train')])
    selected=pd.concat(parts).sort_values(['split','label','doc_id']).reset_index(drop=True)
    validate_manifest(selected)
    if set(selected[selected.split=='test'].source_domain)&set(selected[selected.split!='test'].source_domain):
        raise ValueError('Publisher leakage into training or validation')
    return selected


def build(source,recipe_path,out):
    if out.exists():raise FileExistsError(out)
    frame,config=load_manifest(source)
    recipe=json.loads(recipe_path.read_text(encoding='utf-8'))
    if recipe['parent_manifest_sha256']!=config['manifest_sha256']:raise ValueError('Parent dataset differs')
    if set(recipe['heldout_sources'])!=set(config['styles']):raise ValueError('Every style needs held-out sources')
    selected=allocate(frame,recipe)
    out.mkdir(parents=True)
    shutil.copytree(source/'tokenizer',out/'tokenizer')
    if (source/'tokenizers').exists():shutil.copytree(source/'tokenizers',out/'tokenizers')
    hashes={}
    for split in ['train','validation','test']:
        path=out/f'{split}.csv';selected.loc[selected.split==split,['text','label']].to_csv(path,index=False,encoding='utf-8')
        hashes[path.name]=file_hash(path)
    clean=selected.astype(object).where(pd.notna(selected),None)
    write_jsonl(out/'manifest.jsonl',clean.to_dict('records'))
    write_jsonl(out/'metadata.jsonl',clean.drop(columns='text').to_dict('records'))
    config={**config,'seed':recipe['seed'],'parent_dataset':str(source),
        'parent_manifest_sha256':config['manifest_sha256'],'manifest_sha256':file_hash(out/'manifest.jsonl'),
        'metadata_sha256':file_hash(out/'metadata.jsonl'),'input_file_hashes':hashes,
        'requested_per_class':sum(recipe[k] for k in ['train_per_style','validation_per_style','test_per_style']),
        'split_counts':{s:p.label.value_counts().sort_index().to_dict() for s,p in selected.groupby('split')},
        'diagnostic_recipe_sha256':file_hash(recipe_path),'evaluation_design':'publisher_holdout_development_diagnostic',
        'limitations':['Provisional labels, not expert gold; parent development documents previously inspected.',
            'Test publishers excluded from train/validation; split groups touching them excluded from train/validation.',
            'Publisher, genre, topic and length can change together; not an isolated causal source effect.',
            'Templates appear in the held-out diagnostic only; this derived run cannot train on them.',
            'Not a final external benchmark and not a basis for deployment model selection.']}
    write_json(out/'config.json',config);write_json(out/'recipe.json',recipe)
    write_json(out/'COMPLETE.json',{'manifest_sha256':config['manifest_sha256']})
    load_manifest(out)
    summary={'documents':len(selected),'manifest_sha256':config['manifest_sha256'],
        'split_counts':selected.groupby('split').size().to_dict(),'heldout_sources':recipe['heldout_sources'],
        'publisher_overlap':0,'split_group_overlap':0,'final_external_test':False}
    write_json(out/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();build(a.source,a.recipe,a.out_dir)
