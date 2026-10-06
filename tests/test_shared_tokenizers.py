import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from kazstyle.models.tokenization import fit_shared_text,load_dataset_tokenizer
from kazstyle.models.style_transformer import StyleTransformer
from kazstyle.data.corpus import file_hash,load_manifest


class ToyTokenizer:
    def __init__(self,multiplier):self.multiplier=multiplier
    def __call__(self,text,**kwargs):return {'input_ids':[1]*(2+len(text.split())*self.multiplier)}


class SharedTokenizerTests(unittest.TestCase):
    def test_same_whole_word_excerpt_fits_both_encoders(self):
        self.assertEqual(fit_shared_text('Бір екі үш төрт',[(ToyTokenizer(1),6),(ToyTokenizer(3),8)]),'Бір екі')
        with self.assertRaises(ValueError):fit_shared_text('Бір',[(ToyTokenizer(3),4)])

    def test_unregistered_or_tampered_tokenizer_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'tokenizer').mkdir();(root/'tokenizer/vocab').write_text('one')
            config={'tokenizer_name':'primary','max_tokens':8,'tokenizer_hashes':{'vocab':file_hash(root/'tokenizer/vocab')}}
            with self.assertRaisesRegex(ValueError,'not uniquely registered'):load_dataset_tokenizer(root,config,'other')
            (root/'tokenizer/vocab').write_text('changed')
            with self.assertRaisesRegex(ValueError,'snapshot'):load_dataset_tokenizer(root,config,'primary')

    def test_token_type_ids_reach_bert_encoder(self):
        class Encoder(torch.nn.Module):
            config=SimpleNamespace(hidden_size=4,num_hidden_layers=4)
            def forward(self,**kwargs):
                self.seen=kwargs
                return SimpleNamespace(last_hidden_state=torch.zeros(1,2,4))
        encoder=Encoder()
        with patch('kazstyle.models.style_transformer.AutoModel.from_config',return_value=encoder):
            model=StyleTransformer('tiny',2,encoder_config=object())
        model(input_ids=torch.ones(1,2,dtype=torch.long),attention_mask=torch.ones(1,2),token_type_ids=torch.ones(1,2))
        self.assertTrue(torch.equal(encoder.seen['token_type_ids'],torch.ones(1,2)))

    def test_shared_dataset_keeps_documents_labels_and_splits(self):
        source=Path('data/processed/text_only_v3');target=Path('data/processed/text_only_v4_shared')
        if not target.exists():self.skipTest('Local shared dataset not present')
        old,_=load_manifest(source);new,config=load_manifest(target)
        self.assertEqual(old[['doc_id','label','split']].to_dict('records'),new[['doc_id','label','split']].to_dict('records'))
        self.assertEqual(sum(old.text!=new.text),3)
        for model in [config['tokenizer_name'],config['additional_tokenizers'][0]['model_name']]:
            tokenizer,spec=load_dataset_tokenizer(target,config,model)
            self.assertLessEqual(max(len(ids) for ids in tokenizer(new.text.tolist(),truncation=False)['input_ids']),spec['max_tokens'])


if __name__=='__main__':unittest.main()
