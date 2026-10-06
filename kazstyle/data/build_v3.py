"""Build versioned text-only splits from quality-screened candidate documents."""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from kazstyle.data.corpus import (STYLES, GLOBAL_IDS, DEFAULT_MODEL, digest, file_hash,
    normalized_hash, make_excerpt, duplicate_groups, balanced_splits, validate_manifest, write_json, write_jsonl)
from kazstyle.data.quality import CLEANING_VERSION, assert_text_only, inspect_text
from kazstyle.settings import project_path
from kazstyle.data.grouping import assign_split_groups, assign_passage_groups, validate_group_separation, GROUP_KEYS
from kazstyle.models.tokenization import load_dataset_tokenizer, tokenizer_specs, fit_shared_text


def diverse_sample(part, n, seed):
    """Round-robin source/genre strata; no duplicating rare examples."""
    buckets=[p.sample(frac=1,random_state=seed).to_dict('records')
             for _,p in part.groupby(['source_domain','genre'],sort=True)]
    chosen=[]
    while buckets and len(chosen)<n:
        next_buckets=[]
        for bucket in buckets:
            if len(chosen)<n:chosen.append(bucket.pop())
            if bucket:next_buckets.append(bucket)
        buckets=next_buckets
    return pd.DataFrame(chosen)


def build(candidates, out, per_class=240, seed=42, styles=None, tokenizer_path=None, shared_reference=None):
    styles=styles or list(STYLES)
    if out.exists():raise FileExistsError(out)
    if not 2<=len(styles)<=5 or len(set(styles))!=len(styles) or any(s not in STYLES for s in styles):
        raise ValueError('Choose 2 to 5 distinct styles')
    if per_class<40:raise ValueError('At least 40 requested documents per class')
    shared_loaded = []
    tokenizer_name = DEFAULT_MODEL
    if shared_reference:
        if tokenizer_path:
            raise ValueError('Choose a shared reference or a tokenizer path, not both')
        reference_config = json.loads((shared_reference/'config.json').read_text(encoding='utf-8'))
        if reference_config['max_tokens'] != 256:
            raise ValueError('This builder expects a primary tokenizer limit of 256')
        shared_loaded = [load_dataset_tokenizer(shared_reference,reference_config,s['model_name'])
                         for s in tokenizer_specs(reference_config)]
        tokenizer = shared_loaded[0][0]; tokenizer_name = reference_config['tokenizer_name']
    else:
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_path or DEFAULT_MODEL,local_files_only=True)
    rows=[json.loads(line) for line in candidates.read_text(encoding='utf-8').splitlines()]
    if any(str(r.get('usage_status','')).startswith('reserved') for r in rows):
        raise ValueError('Reserved source/evaluation candidates cannot be included in training datasets')
    mapping={s:i for i,s in enumerate(styles)}
    rows=[r for r in rows if r['style'] in styles]
    available=Counter(r['style'] for r in rows)
    if any(available[s]<40 for s in styles):raise ValueError(f'Insufficient screened data: {available}')
    raw=pd.DataFrame(rows)
    # Cap the pairwise near-duplicate workload; retain the full clean candidate store.
    raw=pd.concat([diverse_sample(p,min(len(p),max(per_class*3,1000)),seed)
                   for _,p in raw.groupby('style')],ignore_index=True)
    records=[];quarantine=[]
    for r in raw.to_dict('records'):
        full=r['text'];genre=r.get('genre','unknown')
        # Short/medium/long views occur in every class; policy fixed before evaluation.
        word_budget=[40,80,160][int(digest(r['doc_id'])[:8],16)%3]
        view = r.get('model_view')
        if isinstance(view, dict):
            if view.get('text_sha256') != digest(full):
                raise ValueError('Frozen model view changed')
            excerpt = full
            token_count = len(tokenizer(excerpt, add_special_tokens=True)['input_ids'])
            if token_count > 256:
                raise ValueError('Frozen review view does not fit dataset tokenizer; do not silently change the reviewed text')
            word_budget = view['word_budget']
            details = {'word_start': 0, 'char_start': 0, 'char_end': len(full),
                       'tokens_before_limit': token_count, 'token_count': token_count,
                       'excerpt_words': len(full.split())}
        else:
            excerpt,details=make_excerpt(full,tokenizer,word_budget,256)
        if shared_loaded:
            shared_text = fit_shared_text(excerpt,[(t,s['max_tokens']) for t,s in shared_loaded])
            if isinstance(view,dict) and shared_text != excerpt:
                raise ValueError('Frozen review view exceeds a registered tokenizer limit')
            excerpt = shared_text
            details.update(char_end=details['char_start']+len(excerpt),
                           token_count=len(tokenizer(excerpt,add_special_tokens=True)['input_ids']),
                           excerpt_words=len(excerpt.split()))
        stats,reasons=inspect_text(excerpt,min_words=8)
        if reasons:
            quarantine.append({'doc_id':r['doc_id'],'reasons':reasons});continue
        assert_text_only([excerpt])
        parent=r.get('parent_id') or r['source_url']
        records.append({**r,'label':mapping[r['style']],'global_label':GLOBAL_IDS[r['style']],
            'style_name':STYLES[r['style']],'text':excerpt,'full_clean_text':full,
            'sample_id':digest(r['doc_id']+'|'+excerpt)[:24],
            'content_hash':r['parent_content_hash'] if isinstance(r.get('parent_content_hash'),str) else normalized_hash(full),
            'excerpt_hash':normalized_hash(excerpt),
            'duplicate_probe':r['duplicate_probe'] if isinstance(r.get('duplicate_probe'),str) else ' '.join(full.split()[:120]+full.split()[-120:]),
            'parent_group':parent,'view_max_words':word_budget,**details})
    frame=pd.DataFrame(records).reset_index(drop=True)
    # Duplicate detection uses parent grouping as well as exact and approximate text.
    probes=frame.copy();probes['doc_id']=frame['parent_group']
    print(f'[v3 dedup] {len(frame)} candidates',flush=True)
    frame['duplicate_group_id'],edges=duplicate_groups(probes,.90)
    conflicts=set(frame.groupby('duplicate_group_id').label.nunique().loc[lambda s:s>1].index)
    for r in frame[frame.duplicate_group_id.isin(conflicts)].to_dict('records'):
        quarantine.append({'doc_id':r['doc_id'],'reasons':['duplicate_label_conflict']})
    frame=frame[~frame.duplicate_group_id.isin(conflicts)].drop_duplicates('doc_id').drop_duplicates('content_hash').drop_duplicates('excerpt_hash')
    frame['passage_group_id'], passage_edges = assign_passage_groups(frame)
    frame['split_group_id'] = assign_split_groups(frame)
    # A single collection of form templates is training material, not an independent test.
    template_genres = {'application_form', 'personnel_order', 'contract', 'service_memo',
                       'explanatory_note', 'business_form', 'order_extract',
                       'administrative_order', 'procurement_order', 'procurement_contract', 'business_letter'}
    form_groups=set(frame.loc[frame.genre.isin(template_genres),'split_group_id'])
    forms=frame[frame.split_group_id.isin(form_groups)].copy()
    main=frame[~frame.split_group_id.isin(form_groups)].copy()
    selected,counts=balanced_splits(main,seed,per_class,group_key='split_group_id')
    if len(forms):
        forms['split']='train'
        # Replace a few legal-act examples, preserving class balance and group separation.
        label=mapping['official'];train_official=selected[(selected.split=='train')&(selected.label==label)]
        take=min(len(forms),max(1,len(train_official)//4))
        selected=selected.drop(train_official.tail(take).index)
        selected=pd.concat([selected,forms.head(take)],ignore_index=True)
    selected=selected.sort_values(['split','label','doc_id']).reset_index(drop=True)
    validate_manifest(selected)
    validate_group_separation(selected)
    for key in ['parent_group','content_hash']:
        if (selected.groupby(key).split.nunique()>1).any():raise ValueError(f'Cross-split overlap: {key}')
    out.mkdir(parents=True);tokenizer.save_pretrained(out/'tokenizer')
    additional_tokenizers=[]
    for index,(extra_tokenizer,spec) in enumerate(shared_loaded[1:],1):
        relative=f'tokenizers/encoder_{index}'
        extra_tokenizer.save_pretrained(out/relative)
        additional_tokenizers.append({**spec,'path':relative,
            'hashes':{p.name:file_hash(p) for p in (out/relative).iterdir() if p.is_file()}})
    # Physical separation: these are the ONLY files supplying inputs to training.
    input_hashes={}
    for split in ['train','validation','test']:
        path=out/f'{split}.csv'
        selected.loc[selected.split==split,['text','label']].to_csv(path,index=False,encoding='utf-8')
        input_hashes[path.name]=file_hash(path)
    metadata_columns=['sample_id','doc_id','parent_group','duplicate_group_id','source_url','source_domain',
                      'style','style_name','label','split','genre','label_origin','review_status','content_hash','excerpt_hash',
                      'excerpt_words','token_count','view_max_words']
    metadata_columns += [key for key in ['split_group_id','author_group','author_kind','work_group','template_family_id','passage_group_id'] if key in selected]
    metadata=selected[metadata_columns].astype(object).where(pd.notna(selected[metadata_columns]),None)
    write_jsonl(out/'metadata.jsonl',metadata.to_dict('records'))
    # Compatibility report manifest is validated against the separate text-only inputs.
    manifest=metadata.assign(text=selected.text)
    write_jsonl(out/'manifest.jsonl',manifest.to_dict('records'))
    write_jsonl(out/'quarantine.jsonl',quarantine)
    write_jsonl(out/'passage_edges.jsonl',passage_edges)
    fixed_views = 'model_view' in selected and all(isinstance(v,dict) for v in selected.model_view)
    view_policies = sorted({v.get('policy','legacy_mixed') for v in selected.model_view if isinstance(v,dict)}) if 'model_view' in selected else []
    config={'schema_version':3,'cleaning_version':CLEANING_VERSION,'input_columns':['text','label'],
        'model_features':['text'],'styles':styles,'label_to_id':mapping,
        'id_to_label':{str(i):STYLES[s] for s,i in mapping.items()},'seed':seed,
        'requested_per_class':per_class,'max_words':160,'max_tokens':256,'min_document_words':8,
        'excerpt_policy':('frozen shared text views; short documents kept whole; longer documents use one central excerpt <=160 words and every registered tokenizer limit' if fixed_views and view_policies==['full_or_central'] else 'one central excerpt per document; deterministic 40/80/160-word training views; <=256 tokens'),
        'frozen_input_views':bool(fixed_views), 'view_policies':view_policies,
        'tokenizer_name':tokenizer_name,'tokenizer_hashes':{p.name:file_hash(p) for p in (out/'tokenizer').iterdir() if p.is_file()},
        'additional_tokenizers':additional_tokenizers,
        'tokenizer_revision':shared_loaded[0][1].get('revision') if shared_loaded else None,
        'manifest_sha256':file_hash(out/'manifest.jsonl'),'input_file_hashes':input_hashes,
        'metadata_sha256':file_hash(out/'metadata.jsonl'),'candidate_sha256':file_hash(candidates),
        'split_grouping_version':'known_author_work_template_passage_v2', 'split_group_keys':list(GROUP_KEYS),
        'passage_grouping':{'audit_version':'word_passage_overlap_v2','width':5,'min_shared':10,
            'containment_threshold':.8,'passage_min_shared':40,'passage_threshold':.2,'short_passage_threshold':.5,
            'edges_sha256':file_hash(out/'passage_edges.jsonl'),
            'policy':'Conservatively keep every flagged shared-passage component in one split; not a duplicate-document judgement.'},
        'split_counts':{s:p.label.value_counts().sort_index().to_dict() for s,p in selected.groupby('split')},
        'limitations':['Source/category labels are automatically screened, not expert gold.',
            'Document/group-disjoint internal test; not a held-out-publisher evaluation.',
            'Telegram conversation IDs unavailable: contiguous 5000-line block proxies may miss related conversations.',
            'Collected business templates are training-only; short-form generalization needs independent tests.',
            'Known authors, works and template families are grouped; missing or ambiguous author identities remain a limitation.',
            'Engineering development dataset; previously used documents may be reassigned. Not an independent final test.',
            'Length and source distributions can still correlate with labels.']}
    summary={'candidate_counts':dict(available),'selected_documents':len(selected),
        'selected_by_style':dict(Counter(selected['style'])),'groups':selected.duplicate_group_id.nunique(),
        'source_by_style':pd.crosstab(selected.source_domain,selected['style']).to_dict(),
        'length_by_style':selected.groupby('style').excerpt_words.agg(['min','median','max']).to_dict('index'),
        'input_address_count':0,'quarantine_count':len(quarantine),
        'split_groups':selected.split_group_id.nunique(),'near_duplicate_edges':len(edges)}
    write_json(out/'config.json',config);write_json(out/'summary.json',summary)
    write_json(out/'COMPLETE.json',{'manifest_sha256':config['manifest_sha256']})
    print(json.dumps(summary,ensure_ascii=True),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--per-class',type=int,default=240);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--styles',nargs='+',default=list(STYLES))
    p.add_argument('--tokenizer-path',type=Path,help='Optional saved tokenizer; defaults to local pretrained cache')
    p.add_argument('--shared-reference',type=Path,help='Reuse verified tokenizer constraints and snapshots from a completed shared dataset')
    a=p.parse_args();build(a.candidates,a.out_dir,a.per_class,a.seed,a.styles,a.tokenizer_path,a.shared_reference)
