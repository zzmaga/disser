"""Train Kaz-RoBERTa on the exact same fixed manifest as classical baselines.

Both the encoder and the classification head are fine-tuned.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from kazstyle.settings import project_path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,DataCollatorWithPadding,get_linear_schedule_with_warmup

from kazstyle.data.corpus import DEFAULT_MODEL,load_manifest,write_json
from kazstyle.evaluation.reports import metrics,provenance,render_report,save_predictions
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
    p.add_argument('--head',choices=['last','concat4'],default='last')
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--allow-download',action='store_true')
    a = p.parse_args()
    if a.out_dir.exists():
        raise FileExistsError(f'Refusing to overwrite run: {a.out_dir}')
    if a.epochs<1 or a.batch_size<1 or a.threads<1:
        raise ValueError('Positive epochs, batch size and threads required')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True,warn_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frame,config = load_manifest(a.dataset)
    if a.model_name!=config['tokenizer_name']:
        raise ValueError('Build a shared dataset for the requested tokenizer before comparing models')
    tokenizer = AutoTokenizer.from_pretrained(a.dataset/'tokenizer',local_files_only=True)
    collator = DataCollatorWithPadding(tokenizer,padding=True)
    parts = {s:frame[frame.split==s].reset_index(drop=True) for s in ['train','validation','test']}
    loaders = {s:DataLoader(encode(part,tokenizer,config['max_tokens']),batch_size=a.batch_size,
                            shuffle=s=='train',collate_fn=collator,
                            generator=torch.Generator().manual_seed(a.seed)) for s,part in parts.items()}
    model = StyleTransformer(a.model_name,len(config['styles']),a.head,not a.allow_download).to(device)
    a.out_dir.mkdir(parents=True)
    tokenizer.save_pretrained(a.out_dir/'tokenizer')
    model.encoder.config.save_pretrained(a.out_dir/'encoder_config')
    params = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    params.update(device=str(device),encoder_commit=getattr(model.encoder.config,'_commit_hash',None))
    write_json(a.out_dir/'provenance.json',provenance(a.dataset,params))
    write_json(a.out_dir/'dataset_config.json',config)
    optimizer = torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    steps = a.epochs*len(loaders['train'])
    scheduler = get_linear_schedule_with_warmup(optimizer,int(steps*.1),steps)
    history,best = [],-1
    start = time.perf_counter()
    print(f'[train] {a.model_name} head={a.head} device={device} epochs={a.epochs} train={len(parts["train"])}',flush=True)
    for epoch in range(1,a.epochs+1):
        model.train();total_loss=0.;n=0
        for step,batch in enumerate(loaders['train'],1):
            y = batch.pop('labels').to(device)
            logits = model(**{k:v.to(device) for k,v in batch.items()})
            loss = torch.nn.functional.cross_entropy(logits,y)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();scheduler.step()
            total_loss+=loss.item()*len(y);n+=len(y)
            if step==1 or step%10==0:
                print(f'epoch={epoch} batch={step}/{len(loaders["train"])} loss={total_loss/n:.4f} elapsed={time.perf_counter()-start:.1f}s',flush=True)
        val_pred = predict(model,loaders['validation'],device)
        validation = metrics(parts['validation'].label,val_pred,config)
        history.append({'epoch':epoch,'train_loss':total_loss/n,'validation':validation})
        write_json(a.out_dir/'history.json',history)
        if validation['macro_f1']>best:
            best=validation['macro_f1']
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'head':a.head,'model_name':a.model_name,'num_labels':len(config['styles']),
                        'epoch':epoch,'manifest_sha256':config['manifest_sha256']},a.out_dir/'best_model.pt')
        print(f'[epoch {epoch}] validation Macro-F1={validation["macro_f1"]:.4f}',flush=True)
    fit_seconds=time.perf_counter()-start
    # Best epoch is selected by validation only; test is evaluated once after training.
    checkpoint=torch.load(a.out_dir/'best_model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state_dict']);model.to(device)
    val_pred=predict(model,loaders['validation'],device)
    pred_start=time.perf_counter();test_pred=predict(model,loaders['test'],device)
    pred_seconds=time.perf_counter()-pred_start
    model_name='kaz_roberta_'+a.head
    result={'validation':metrics(parts['validation'].label,val_pred,config),
            'test':metrics(parts['test'].label,test_pred,config),'fit_seconds':fit_seconds,
            'test_predict_seconds':pred_seconds,'best_epoch':checkpoint['epoch'],
            'test_ms_per_document':1000*pred_seconds/len(parts['test'])}
    write_json(a.out_dir/'results.json',{model_name:result})
    save_predictions(a.out_dir/'test_predictions.jsonl',parts['test'],test_pred,config)
    save_predictions(a.out_dir/'validation_predictions.jsonl',parts['validation'],val_pred,config)
    render_report(a.out_dir,frame,config,{model_name:result},title='Kaz-RoBERTa pilot')
    print(json.dumps(result,ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()
