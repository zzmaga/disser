"""One controlled development ablation: shorten train views, keep evaluation fixed."""
import argparse,json,shutil
from pathlib import Path
import pandas as pd
from kazstyle.data.corpus import load_manifest,make_excerpt,digest,normalized_hash,file_hash,write_json,write_jsonl
from kazstyle.data.grouping import assign_passage_groups
from kazstyle.data.quality import inspect_text,assert_text_only
from kazstyle.models.tokenization import load_dataset_tokenizer


def shorten(frame,tokenizer,config,budgets):
    changed=frame.copy(deep=True);audit=[]
    for index,row in frame[frame.split=='train'].iterrows():
        budget=budgets[int(digest(row.doc_id)[:8],16)%len(budgets)]
        text,details=make_excerpt(row.text,tokenizer,budget,config['max_tokens'])
        _,reasons=inspect_text(text,min_words=8)
        if reasons or text==row.text:continue
        assert_text_only([text]);changed.at[index,'text']=text
        changed.at[index,'sample_id']=digest(row.doc_id+'|'+text)[:24]
        changed.at[index,'excerpt_hash']=normalized_hash(text)
        changed.at[index,'excerpt_words']=len(text.split());changed.at[index,'token_count']=details['token_count']
        changed.at[index,'view_max_words']=budget
        audit.append({'doc_id':row.doc_id,'old_sha256':digest(row.text),'new_sha256':digest(text),'old_words':len(row.text.split()),'new_words':len(text.split()),'budget':budget})
    # Shortening can expose new containment with a frozen evaluation view.
    # Revert affected train rows, never alter validation/test or split assignment.
    reverted=set()
    while True:
        _,edges=assign_passage_groups(changed)
        splits=dict(zip(changed.doc_id,changed.split))
        cross=[e for e in edges if splits[e['left_doc_id']]!=splits[e['right_doc_id']]]
        bad={e[k] for e in cross for k in ('left_doc_id','right_doc_id') if splits[e[k]]=='train'}
        if not bad:
            if cross:raise ValueError('Existing validation/test shared passage')
            break
        fresh=bad-reverted
        if not fresh:raise ValueError('The original parent has unresolved shared passages')
        mask=changed.doc_id.isin(fresh);changed.loc[mask,:]=frame.loc[mask,:]
        reverted.update(fresh)
    for item in audit:item['reverted_for_cross_split_passage']=item['doc_id'] in reverted
    for split in ['validation','test']:
        assert changed[changed.split==split].equals(frame[frame.split==split])
    return changed,audit


def build(source,recipe_path,out):
    if out.exists():raise FileExistsError(out)
    frame,config=load_manifest(source);recipe=json.loads(recipe_path.read_text(encoding='utf-8'))
    if config['manifest_sha256']!=recipe['parent_manifest_sha256']:raise ValueError('Parent differs from recipe')
    budgets=recipe['train_word_budgets']
    if not budgets or any(type(n)!=int or not 8<=n<=config['max_words'] for n in budgets):raise ValueError('Invalid budgets')
    tokenizer,_=load_dataset_tokenizer(source,config,config['tokenizer_name'])
    changed,audit=shorten(frame,tokenizer,config,budgets)
    out.mkdir(parents=True);shutil.copytree(source/'tokenizer',out/'tokenizer')
    if (source/'tokenizers').exists():shutil.copytree(source/'tokenizers',out/'tokenizers')
    hashes={}
    for split in ['train','validation','test']:
        path=out/f'{split}.csv';changed.loc[changed.split==split,['text','label']].to_csv(path,index=False,encoding='utf-8')
        hashes[path.name]=file_hash(path)
        if split!='train' and hashes[path.name]!=config['input_file_hashes'][path.name]:raise ValueError('Evaluation CSV changed')
    clean=changed.astype(object).where(pd.notna(changed),None)
    write_jsonl(out/'manifest.jsonl',clean.to_dict('records'));write_jsonl(out/'metadata.jsonl',clean.drop(columns='text').to_dict('records'))
    write_jsonl(out/'train_changes.jsonl',audit)
    config={**config,'manifest_sha256':file_hash(out/'manifest.jsonl'),'metadata_sha256':file_hash(out/'metadata.jsonl'),
        'input_file_hashes':hashes,'parent_dataset':str(source),'parent_manifest_sha256':config['manifest_sha256'],
        'diagnostic_recipe_sha256':file_hash(recipe_path),'evaluation_design':'fixed_v6_evaluation_train_length_ablation',
        'excerpt_policy':'Train-only deterministic 40/80/160-word central views; validation/test and inference retain the parent v6 limits.',
        'limitations':config['limitations']+['Post-hoc development ablation motivated by a known short-application regression; not independent confirmation.']}
    write_json(out/'config.json',config);write_json(out/'recipe.json',recipe)
    write_json(out/'COMPLETE.json',{'manifest_sha256':config['manifest_sha256']})
    load_manifest(out)
    summary={'documents':len(changed),'changed_train_texts':sum(not r['reverted_for_cross_split_passage'] for r in audit),
        'reverted_candidates':sum(r['reverted_for_cross_split_passage'] for r in audit),'validation_and_test_csv_bytes_unchanged':True,
        'manifest_sha256':config['manifest_sha256'],'new_rows_added':0}
    write_json(out/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();build(a.source,a.recipe,a.out_dir)
