"""Discarded train-only CUDA smoke steps; no evaluation or saved model weights."""
import argparse
import json
import time
from pathlib import Path

import torch

from kazstyle.data.corpus import load_manifest,write_json
from kazstyle.models.style_transformer import StyleTransformer
from kazstyle.models.tokenization import load_dataset_tokenizer


def check(dataset,model_name,head,revision,out,steps=3,fused=False):
    if out.exists():raise FileExistsError(out)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable in this Python environment')
    if not 2<=steps<=10:raise ValueError('Smoke test must use 2 to 10 steps')
    torch.set_num_threads(2);torch.manual_seed(42)
    frame,config=load_manifest(dataset)
    tokenizer,spec=load_dataset_tokenizer(dataset,config,model_name)
    if spec.get('revision') and revision!=spec['revision']:raise ValueError('Wrong registered encoder revision')
    train=frame[frame.split=='train'].copy()
    train['probe_tokens']=[len(ids) for ids in tokenizer(train.text.tolist(),truncation=False)['input_ids']]
    sample=train.sort_values('probe_tokens',ascending=False).iloc[0]
    encoded=tokenizer(sample.text,return_tensors='pt',truncation=False)
    if encoded['input_ids'].shape[1]>spec['max_tokens']:raise ValueError('Input exceeds registered limit')
    model=StyleTransformer(model_name,len(config['styles']),head,local_only=True,revision=revision)
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
            'manifest_sha256':config['manifest_sha256'],'weights_saved':False}
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
    a=p.parse_args();check(a.dataset,a.model_name,a.head,a.revision,a.out_file,a.steps,a.fused_adamw)
