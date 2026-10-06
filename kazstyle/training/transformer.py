"""Train a registered pretrained encoder on the same fixed texts as all baselines.

Both the encoder and the classification head are fine-tuned.
"""
from __future__ import annotations

import argparse
import json
import random
import math
import time
from contextlib import nullcontext
from pathlib import Path
from kazstyle.settings import project_path, require_new_run

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,DataCollatorWithPadding,get_linear_schedule_with_warmup

from kazstyle.data.corpus import DEFAULT_MODEL,load_manifest,write_json,file_hash
from kazstyle.models.tokenization import load_dataset_tokenizer
from kazstyle.evaluation.reports import metrics,save_provenance,render_report,save_predictions
from kazstyle.models.style_transformer import StyleTransformer


def encode(part,tokenizer,max_tokens):
    encoded = tokenizer(part.text.tolist(),truncation=False,padding=False)
    if max(map(len,encoded['input_ids']))>max_tokens:
        raise ValueError('Input would be truncated: rebuild the shared dataset for this tokenizer')
    return [{**{key:value[i] for key,value in encoded.items()},'labels':int(label)}
            for i,label in enumerate(part.label)]


@torch.no_grad()
def predict(model,loader,device):
    model.eval()
    predictions = []
    for batch in loader:
        batch.pop('labels')
        logits = model(**{k:v.to(device) for k,v in batch.items()})
        predictions.extend(logits.argmax(-1).cpu().tolist())
    return predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=project_path('data/processed/pilot_v2'))
    p.add_argument('--out-dir',type=Path,default=project_path('artifacts/pilot_v2_roberta_last'))
    p.add_argument('--model-name',default=DEFAULT_MODEL)
    p.add_argument('--revision',help='Pinned encoder revision; must match a registered alternate tokenizer')
    p.add_argument('--head',choices=['last','concat4'],default='last')
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--gradient-accumulation',type=int,default=1)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--precision',choices=['float32','float16'],default='float32')
    p.add_argument('--gradient-checkpointing',action='store_true')
    p.add_argument('--fused-adamw',action='store_true')
    p.add_argument('--allow-download',action='store_true')
    a = p.parse_args()
    require_new_run(a.out_dir)
    if a.epochs<1 or a.batch_size<1 or a.threads<1 or a.gradient_accumulation<1:
        raise ValueError('Positive epochs, batch size and threads required')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True,warn_only=True)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable in this environment')
    if device.type!='cuda' and (a.precision=='float16' or a.fused_adamw):
        raise ValueError('float16 and fused AdamW require CUDA')
    frame,config = load_manifest(a.dataset)
    tokenizer,tokenizer_spec = load_dataset_tokenizer(a.dataset,config,a.model_name)
    if a.revision and tokenizer_spec.get('revision') and a.revision!=tokenizer_spec['revision']:
        raise ValueError('Encoder revision differs from registered tokenizer')
    revision=a.revision or tokenizer_spec.get('revision')
    max_tokens=tokenizer_spec['max_tokens']
    collator = DataCollatorWithPadding(tokenizer,padding=True)
    parts = {s:frame[frame.split==s].reset_index(drop=True) for s in ['train','validation','test']}
    loaders = {s:DataLoader(encode(part,tokenizer,max_tokens),batch_size=a.batch_size,
                            shuffle=s=='train',collate_fn=collator,
                            generator=torch.Generator().manual_seed(a.seed)) for s,part in parts.items()}
    model = StyleTransformer(a.model_name,len(config['styles']),a.head,not a.allow_download,revision=revision).to(device)
    if a.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        if hasattr(model.encoder.config,'use_cache'):model.encoder.config.use_cache=False
    a.out_dir.mkdir(parents=True)
    tokenizer.save_pretrained(a.out_dir/'tokenizer')
    tokenizer_hashes={p.name:file_hash(p) for p in (a.out_dir/'tokenizer').iterdir() if p.is_file()}
    model.encoder.config.save_pretrained(a.out_dir/'encoder_config')
    params = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    params.update(device=str(device),requested_device=a.device,encoder_commit=getattr(model.encoder.config,'_commit_hash',None),
                  max_tokens=max_tokens,effective_batch_size=a.batch_size*a.gradient_accumulation,
                  parameter_count=sum(p.numel() for p in model.parameters()))
    save_provenance(a.out_dir,a.dataset,params)
    write_json(a.out_dir/'dataset_config.json',config)
    optimizer = torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01,
                                 **({'fused':True,'foreach':False} if a.fused_adamw else {}))
    scaler=torch.amp.GradScaler('cuda',init_scale=128.) if a.precision=='float16' else None
    context=lambda:torch.autocast(device_type='cuda',dtype=torch.float16) if scaler is not None else nullcontext()
    steps = a.epochs*math.ceil(len(loaders['train'])/a.gradient_accumulation)
    scheduler = get_linear_schedule_with_warmup(optimizer,int(steps*.1),steps)
    history,best = [],-1
    optimizer_steps=0;skipped_optimizer_steps=0
    start = time.perf_counter()
    print(f'[train] {a.model_name} head={a.head} device={device} epochs={a.epochs} train={len(parts["train"])}',flush=True)
    for epoch in range(1,a.epochs+1):
        model.train();total_loss=0.;n=0
        optimizer.zero_grad(set_to_none=True)
        for step,batch in enumerate(loaders['train'],1):
            y = batch.pop('labels').to(device)
            with context():
                logits = model(**{k:v.to(device) for k,v in batch.items()})
                loss = torch.nn.functional.cross_entropy(logits,y)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            group_start=((step-1)//a.gradient_accumulation)*a.batch_size*a.gradient_accumulation
            group_size=min(a.batch_size*a.gradient_accumulation,len(parts['train'])-group_start)
            weighted_loss=loss*len(y)/group_size
            if scaler is None:weighted_loss.backward()
            else:scaler.scale(weighted_loss).backward()
            if step%a.gradient_accumulation==0 or step==len(loaders['train']):
                if scaler is not None:scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if scaler is None:
                    optimizer.step();updated=True
                else:
                    previous_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                    updated=scaler.get_scale()>=previous_scale
                if updated:scheduler.step();optimizer_steps+=1
                else:skipped_optimizer_steps+=1
                optimizer.zero_grad(set_to_none=True)
            total_loss+=loss.item()*len(y);n+=len(y)
            if step==1 or step%10==0:
                print(f'epoch={epoch} batch={step}/{len(loaders["train"])} loss={total_loss/n:.4f} elapsed={time.perf_counter()-start:.1f}s',flush=True)
        val_pred = predict(model,loaders['validation'],device)
        validation = metrics(parts['validation'].label,val_pred,config)
        if optimizer_steps==0:raise RuntimeError('No optimizer update succeeded')
        history.append({'epoch':epoch,'train_loss':total_loss/n,'validation':validation,
                        'optimizer_steps':optimizer_steps,'skipped_optimizer_steps':skipped_optimizer_steps})
        write_json(a.out_dir/'history.json',history)
        if validation['macro_f1']>best:
            best=validation['macro_f1']
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'head':a.head,'model_name':a.model_name,'num_labels':len(config['styles']),
                        'epoch':epoch,'manifest_sha256':config['manifest_sha256'],
                        'tokenizer_hashes':tokenizer_hashes,'max_tokens':max_tokens},a.out_dir/'best_model.pt')
        print(f'[epoch {epoch}] validation Macro-F1={validation["macro_f1"]:.4f}',flush=True)
    fit_seconds=time.perf_counter()-start
    # Best epoch is selected by validation only; test is evaluated once after training.
    checkpoint=torch.load(a.out_dir/'best_model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state_dict']);model.to(device)
    val_pred=predict(model,loaders['validation'],device)
    pred_start=time.perf_counter();test_pred=predict(model,loaders['test'],device)
    pred_seconds=time.perf_counter()-pred_start
    family='kaz_roberta' if a.model_name==DEFAULT_MODEL else ('mbert' if 'bert-base-multilingual' in a.model_name else a.model_name.rsplit('/',1)[-1])
    model_name=family+'_'+a.head
    result={'validation':metrics(parts['validation'].label,val_pred,config),
            'test':metrics(parts['test'].label,test_pred,config),'fit_seconds':fit_seconds,
            'test_predict_seconds':pred_seconds,'best_epoch':checkpoint['epoch'],
            'test_ms_per_document':1000*pred_seconds/len(parts['test'])}
    write_json(a.out_dir/'results.json',{model_name:result})
    save_predictions(a.out_dir/'test_predictions.jsonl',parts['test'],test_pred,config)
    save_predictions(a.out_dir/'validation_predictions.jsonl',parts['validation'],val_pred,config)
    render_report(a.out_dir,frame,config,{model_name:result},title=a.model_name+' · '+a.head)
    print(json.dumps(result,ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()
