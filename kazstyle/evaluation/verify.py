"""Verify a trained checkpoint can be restored offline from its saved config."""
import argparse
import json
from pathlib import Path
from kazstyle.settings import project_path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig,AutoModel,AutoTokenizer,DataCollatorWithPadding

from kazstyle.data.corpus import load_manifest,write_json
from kazstyle.models.style_transformer import StyleTransformer
from kazstyle.training.transformer import encode,predict


def verify(run,dataset):
    torch.set_num_threads(2)
    frame,config=load_manifest(dataset)
    checkpoint=torch.load(run/'best_model.pt',map_location='cpu',weights_only=True)
    if checkpoint['manifest_sha256']!=config['manifest_sha256']:
        raise ValueError('Wrong dataset')
    # Bypass pretrained downloading/initialization: all learned tensors are in the checkpoint.
    encoder_config=AutoConfig.from_pretrained(run/'encoder_config',local_files_only=True)
    model=StyleTransformer.__new__(StyleTransformer)
    torch.nn.Module.__init__(model)
    model.encoder=AutoModel.from_config(encoder_config)
    model.head=checkpoint['head']
    width=encoder_config.hidden_size*(4 if model.head=='concat4' else 1)
    model.dropout=torch.nn.Dropout(.1)
    model.classifier=torch.nn.Linear(width,checkpoint['num_labels'])
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    tokenizer=AutoTokenizer.from_pretrained(run/'tokenizer',local_files_only=True)
    part=frame[frame.split=='test'].iloc[:12]
    loader=DataLoader(encode(part,tokenizer,config['max_tokens']),batch_size=4,
                      collate_fn=DataCollatorWithPadding(tokenizer))
    actual=predict(model,loader,torch.device('cpu'))
    rows=[json.loads(x) for x in (run/'test_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    expected={row['sample_id']:row['y_pred'] for row in rows}
    wanted=[expected[s] for s in part.sample_id]
    if not np.array_equal(actual,wanted):raise ValueError('Reload changed predictions')
    result={'offline_reload_verified':True,'checked_samples':len(part),
            'strict_state_dict_load':True,'manifest_sha256':config['manifest_sha256']}
    write_json(run/'reload_verification.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=project_path('data/processed/pilot_v2'))
    a=p.parse_args();verify(a.run_dir,a.dataset)
