"""Verified tokenizer snapshots and a shared text boundary across encoders."""
import re
from transformers import AutoTokenizer
from kazstyle.data.corpus import file_hash


def tokenizer_specs(config):
    return [{'model_name':config['tokenizer_name'],'path':'tokenizer',
             'max_tokens':config['max_tokens'],'hashes':config['tokenizer_hashes'],
             'revision':config.get('tokenizer_revision')},*config.get('additional_tokenizers',[])]


def load_dataset_tokenizer(dataset,config,model_name):
    specs=[s for s in tokenizer_specs(config) if s['model_name']==model_name]
    if len(specs)!=1:raise ValueError('Model tokenizer is not uniquely registered in the shared dataset')
    spec=specs[0];path=dataset/spec['path']
    for name,sha in spec['hashes'].items():
        if file_hash(path/name)!=sha:raise ValueError('Tokenizer differs from frozen dataset snapshot')
    return AutoTokenizer.from_pretrained(path,local_files_only=True),spec


def fit_shared_text(text,constraints):
    """Remove whole trailing words only; never silently truncate encoder input."""
    while text:
        if all(len(tokenizer(text,add_special_tokens=True,truncation=False,verbose=False)['input_ids'])<=limit
               for tokenizer,limit in constraints):return text
        words=list(re.finditer(r'\S+',text))
        text=text[:words[-2].end()] if len(words)>1 else ''
    raise ValueError('No complete word fits all registered tokenizers')
