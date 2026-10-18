"""Discarded train-only CUDA smoke steps; no evaluation or saved model weights."""
import argparse
import json
import time
from pathlib import Path

import torch

from kazstyle.data.corpus import load_manifest,write_json,file_hash,digest
from kazstyle.models.style_transformer import StyleTransformer
from kazstyle.models.tokenization import load_dataset_tokenizer,tokenizer_specs,fit_shared_text
from kazstyle.data.reservation import assert_training_allowed, protected_rows


def check(dataset,model_name,head,revision,out,steps=3,fused=False,*,
          tokenizer_reference=None,eligible_only=False,allow_download=False,fit_reference_view=False):
    if out.exists():raise FileExistsError(out)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable in this Python environment')
    if not 2<=steps<=10:raise ValueError('Smoke test must use 2 to 10 steps')
    torch.set_num_threads(2);torch.manual_seed(42)
    frame,config=load_manifest(dataset)
    reference=tokenizer_reference or dataset
    reference_config=json.loads((reference/'config.json').read_text(encoding='utf-8'))
    tokenizer,spec=load_dataset_tokenizer(reference,reference_config,model_name)
    if spec.get('revision') and revision!=spec['revision']:raise ValueError('Wrong registered encoder revision')
    train=frame[frame.split=='train'].copy()
    excluded=0
    if eligible_only:
        blocked=protected_rows(train.to_dict('records'))
        train=train.iloc[[i for i in range(len(train)) if i not in blocked]].copy()
        excluded=len(blocked)
    reservations=assert_training_allowed(train.to_dict('records'))
    if train.empty:raise ValueError('No unreserved train examples remain for a hardware probe')
    train['original_text_sha256']=[digest(text) for text in train.text]
    changed=0
    if fit_reference_view:
        if tokenizer_reference is None:raise ValueError('A separate tokenizer reference is required for projected probe views')
        constraints=[(tok,spec['max_tokens']) for tok,spec in
            (load_dataset_tokenizer(reference,reference_config,s['model_name'])
             for s in tokenizer_specs(reference_config))]
        projected=[fit_shared_text(text,constraints) for text in train.text]
        changed=sum(left!=right for left,right in zip(train.text,projected))
        train['text']=projected
    train['probe_tokens']=[len(ids) for ids in tokenizer(train.text.tolist(),truncation=False)['input_ids']]
    sample=train.sort_values('probe_tokens',ascending=False).iloc[0]
    encoded=tokenizer(sample.text,return_tensors='pt',truncation=False)
    if encoded['input_ids'].shape[1]>spec['max_tokens']:raise ValueError('Input exceeds registered limit')
    model=StyleTransformer(model_name,len(config['styles']),head,local_only=not allow_download,revision=revision)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if hasattr(model.encoder.config,'use_cache'):model.encoder.config.use_cache=False
    model.cuda();model.train()
    encoded={k:v.cuda() for k,v in encoded.items()};label=torch.tensor([int(sample.label)],device='cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01,foreach=False,fused=fused)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    torch.cuda.reset_peak_memory_stats();timings=[];losses=[];scales=[];updates=[]
    try:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
            previous=model.classifier.weight.detach().clone()
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                logits=model(**encoded);loss=torch.nn.functional.cross_entropy(logits,label)
            if not torch.isfinite(loss):raise ValueError('Non-finite smoke loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            updates.append(not torch.equal(previous,model.classifier.weight))
            timings.append(time.perf_counter()-start);losses.append(float(loss.detach()))
            scales.append(scaler.get_scale())
            print(json.dumps({'smoke_step':len(timings),'loss':losses[-1],'scale':scales[-1],
                              'updated':updates[-1],'seconds':timings[-1]}),flush=True)
        if not any(updates):raise ValueError('No optimizer update succeeded')
        result={'status':'passed','purpose':'Discarded train-only hardware smoke check; NOT an accuracy experiment',
            'model_name':model_name,'head':head,'encoder_revision':revision,
            'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
            'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'microbatch':1,'mixed_precision':'float16_with_GradScaler','gradient_checkpointing':True,
            'adamw_foreach':False,'adamw_fused':fused,'steps':steps,'step_seconds':timings,'losses':losses,'scales':scales,'optimizer_updated':updates,
            'sample_id':sample.sample_id,'sample_tokens':int(sample.probe_tokens),'sample_split':'train',
            'manifest_sha256':config['manifest_sha256'],'weights_saved':False,
            'evaluation_reservations':reservations,'excluded_reserved_train_rows':excluded,
            'tokenizer_reference':str(reference),'tokenizer_config_sha256':file_hash(reference/'config.json'),
            'encoder_resolved_commit':getattr(model.encoder.config,'_commit_hash',None),
            'probe_view_projected_to_reference':fit_reference_view,'projected_train_views':changed,
            'sample_original_text_sha256':sample.original_text_sha256,
            'sample_probe_text_sha256':digest(sample.text),
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'source_sha256':{str(Path(__file__)):file_hash(Path(__file__))}}
        out.parent.mkdir(parents=True,exist_ok=True);write_json(out,result);print(json.dumps(result))
    except torch.cuda.OutOfMemoryError:
        out.parent.mkdir(parents=True,exist_ok=True)
        write_json(out,{'status':'out_of_memory','model_name':model_name,'head':head,
                       'purpose':'Hardware smoke check only','weights_saved':False})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--model-name',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--head',choices=['last','concat4'],default='last');p.add_argument('--out-file',type=Path,required=True)
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--fused-adamw',action='store_true')
    p.add_argument('--tokenizer-reference',type=Path,help='Frozen tokenizer-only reference for a new encoder')
    p.add_argument('--eligible-only',action='store_true',help='Discard reserved rows from the legacy TRAIN pool before selecting one smoke input')
    p.add_argument('--allow-download',action='store_true')
    p.add_argument('--fit-reference-view',action='store_true',help='Project eligible legacy TRAIN inputs to every new tokenizer limit for this discarded hardware probe only')
    a=p.parse_args();check(a.dataset,a.model_name,a.head,a.revision,a.out_file,a.steps,a.fused_adamw,
        tokenizer_reference=a.tokenizer_reference,eligible_only=a.eligible_only,allow_download=a.allow_download,
        fit_reference_view=a.fit_reference_view)
