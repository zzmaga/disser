"""Derive shared excerpts for another tokenizer without changing document splits."""
import argparse
import json
import shutil
from pathlib import Path
from transformers import AutoTokenizer
from kazstyle.data.corpus import load_manifest,validate_manifest,digest,normalized_hash,file_hash,write_json,write_jsonl
from kazstyle.models.tokenization import load_dataset_tokenizer,fit_shared_text


def build(source,out,model_source):
    if out.exists():raise FileExistsError(out)
    frame,config=load_manifest(source)
    primary,spec=load_dataset_tokenizer(source,config,config['tokenizer_name'])
    extra_spec=json.loads(model_source.read_text(encoding='utf-8'))
    secondary=AutoTokenizer.from_pretrained(extra_spec['model_name'],revision=extra_spec['revision'],local_files_only=True)
    constraints=[(primary,config['max_tokens']),(secondary,extra_spec['max_tokens'])]
    changes=[]
    for index,row in frame.iterrows():
        text=fit_shared_text(row.text,constraints)
        if text!=row.text:
            changes.append({'doc_id':row.doc_id,'split':row.split,'old_sample_id':row.sample_id,
                            'old_words':len(row.text.split()),'new_words':len(text.split())})
            frame.at[index,'sample_id']=digest(row.doc_id+'|'+text)[:24]
            frame.at[index,'text']=text;frame.at[index,'excerpt_hash']=normalized_hash(text)
            frame.at[index,'excerpt_words']=len(text.split())
            frame.at[index,'token_count']=len(primary(text)['input_ids'])
    validate_manifest(frame)
    out.mkdir(parents=True)
    shutil.copytree(source/'tokenizer',out/'tokenizer')
    secondary.save_pretrained(out/'tokenizers/mbert')
    extra={'model_name':extra_spec['model_name'],'revision':extra_spec['revision'],
           'path':'tokenizers/mbert','max_tokens':extra_spec['max_tokens'],
           'hashes':{p.name:file_hash(p) for p in (out/'tokenizers/mbert').iterdir() if p.is_file()}}
    hashes={}
    for split in ['train','validation','test']:
        path=out/f'{split}.csv';frame.loc[frame.split==split,['text','label']].to_csv(path,index=False,encoding='utf-8')
        hashes[path.name]=file_hash(path)
    write_jsonl(out/'manifest.jsonl',frame.to_dict('records'))
    write_jsonl(out/'metadata.jsonl',frame.drop(columns='text').to_dict('records'))
    config={**config,'additional_tokenizers':[extra],
            'manifest_sha256':file_hash(out/'manifest.jsonl'),'metadata_sha256':file_hash(out/'metadata.jsonl'),
            'input_file_hashes':hashes,'parent_dataset':str(source),
            'parent_manifest_sha256':file_hash(source/'manifest.jsonl'),
            'excerpt_policy':config['excerpt_policy']+'; remove trailing whole words to fit mBERT <=512 tokens for ALL models'}
    write_json(out/'config.json',config)
    write_json(out/'derivation.json',{'changes':changes,'documents':len(frame),'document_splits_unchanged':True,
                                    'labels_unchanged':True,'selection_uses_model_predictions':False})
    write_json(out/'COMPLETE.json',{'manifest_sha256':config['manifest_sha256']})
    load_manifest(out)
    print(json.dumps({'documents':len(frame),'changed_excerpts':changes,'manifest_sha256':config['manifest_sha256']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--model-source',type=Path,required=True)
    a=p.parse_args();build(a.source,a.out_dir,a.model_source)
