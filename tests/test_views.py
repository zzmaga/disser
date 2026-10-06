import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kazstyle.data.corpus import digest,write_json,write_jsonl
from kazstyle.data.views import prepare

TEXT = 'Күздің салқын желі жапырақтарды баяу ұшырып, айналаға сары алтындай шашып жатты. Алыстан мұнартып көрінген тау сілемдері батқан күннің қызыл шапағына шомылып, тылсым бір күйге енгендей. Жүректі тербеген сағыныш пен үнсіз тыныштық даланы ерекше бір сырға бөледі.'


class WordTokenizer:
    def __call__(self,text,**kwargs):
        return {'input_ids':list(range(len(text.split())+2))}


class FrozenViewTests(unittest.TestCase):
    def test_complete_short_document_and_reserved_role_survive(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); reference=root/'reference';reference.mkdir()
            source=root/'input.jsonl'
            write_json(reference/'config.json',{})
            write_jsonl(source,[{'doc_id':'a','text':TEXT,'style':'literary','source_domain':'example.org',
                               'parent_id':'parent','author_group':'writer',
                               'usage_status':'reserved_source_NOT_for_training_or_prediction'}])
            spec={'model_name':'test_tokenizer','max_tokens':256,'hashes':{}}
            with patch('kazstyle.data.views.tokenizer_specs',return_value=[spec]),patch('kazstyle.data.views.load_dataset_tokenizer',return_value=(WordTokenizer(),spec)):
                prepare([source],reference,root/'out')
            row=json.loads((root/'out/candidates.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(row['text'],TEXT)
            self.assertFalse(row['model_view']['is_excerpt'])
            self.assertEqual(row['model_view']['text_sha256'],digest(TEXT))
            self.assertEqual(row['model_view']['word_budget'],160)
            self.assertEqual(row['author_group'],'writer')
            self.assertTrue(row['usage_status'].startswith('reserved'))


if __name__=='__main__':unittest.main()
