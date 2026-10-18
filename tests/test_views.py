import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kazstyle.data.corpus import digest,file_hash,write_json,write_jsonl
from kazstyle.data.views import prepare

TEXT = 'Күздің салқын желі жапырақтарды баяу ұшырып, айналаға сары алтындай шашып жатты. Алыстан мұнартып көрінген тау сілемдері батқан күннің қызыл шапағына шомылып, тылсым бір күйге енгендей. Жүректі тербеген сағыныш пен үнсіз тыныштық даланы ерекше бір сырға бөледі.'


class WordTokenizer:
    def __call__(self,text,**kwargs):
        words=list(re.finditer(r'\S+',text))
        if kwargs.get('truncation'):words=words[:kwargs['max_length']-2]
        result={'input_ids':list(range(len(words)+2))}
        if kwargs.get('return_offsets_mapping'):
            result['offset_mapping']=[(0,0),*(w.span() for w in words),(0,0)]
        return result


class FrozenViewTests(unittest.TestCase):
    def test_development_views_exclude_registered_evaluation_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);corpus,reference=self.canonical_fixture(root)
            spec={'model_name':'test_tokenizer','max_tokens':256,'hashes':{}}
            with patch('kazstyle.data.views.tokenizer_specs',return_value=[spec]),patch('kazstyle.data.views.load_dataset_tokenizer',return_value=(WordTokenizer(),spec)),patch('kazstyle.data.views.protected_rows',return_value={0:['test:identity_or_group']}):
                summary=prepare([],reference,root/'out',corpus=corpus,development_only=True)
            self.assertEqual(summary['views'],0)
            rejected=json.loads((root/'out/quarantine.jsonl').read_text(encoding='utf8'))
            self.assertEqual(rejected['reason'],'evaluation_reservation')

    def canonical_fixture(self,root,text=TEXT):
        from kazstyle.data.export_corpus import build
        source=root/'source.jsonl';write_jsonl(source,[{
            'doc_id':'a','text':text,'style':'literary','source_domain':'example.org',
            'source_url':'https://example.org/a','parent_id':'a','author_group':'writer'}])
        recipe=root/'recipe.json';write_json(recipe,{'purpose':'provisional_document_corpus',
            'inputs':[{'path':str(source),'sha256':file_hash(source)}],'passage_grouping':True})
        corpus=root/'corpus';build(recipe,corpus)
        reference=root/'reference';reference.mkdir();write_json(reference/'config.json',{})
        return corpus,reference

    def test_excerpt_hash_describes_shown_text_and_retains_full_document_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);full=' '.join([TEXT]*3)
            corpus,reference=self.canonical_fixture(root,full)
            spec={'model_name':'test_tokenizer','max_tokens':50,'hashes':{}}
            with patch('kazstyle.data.views.tokenizer_specs',return_value=[spec]),patch('kazstyle.data.views.load_dataset_tokenizer',return_value=(WordTokenizer(),spec)):
                prepare([],reference,root/'out',corpus=corpus)
            row=json.loads((root/'out/candidates.jsonl').read_text(encoding='utf8'))
            self.assertTrue(row['model_view']['is_excerpt'])
            self.assertEqual(row['text_sha256'],digest(row['text']))
            self.assertEqual(row['corpus_text_sha256'],digest(full))
            self.assertNotEqual(row['text_sha256'],row['corpus_text_sha256'])

    def test_canonical_full_document_group_survives_view_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);corpus,reference=self.canonical_fixture(root)
            spec={'model_name':'test_tokenizer','max_tokens':256,'hashes':{}}
            with patch('kazstyle.data.views.tokenizer_specs',return_value=[spec]),patch('kazstyle.data.views.load_dataset_tokenizer',return_value=(WordTokenizer(),spec)):
                prepare([],reference,root/'out',corpus=corpus)
            original=json.loads((corpus/'metadata.jsonl').read_text(encoding='utf8'))
            row=json.loads((root/'out/candidates.jsonl').read_text(encoding='utf8'))
            self.assertEqual(row['corpus_split_group_id'],original['split_group_id'])
            self.assertEqual(row['corpus_text_sha256'],digest(TEXT))
            self.assertEqual(row['text'],TEXT)
            with self.assertRaisesRegex(ValueError,'Choose JSONL'):
                prepare([root/'source.jsonl'],reference,root/'both',corpus=corpus)

    def test_corpus_changed_after_loading_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);corpus,reference=self.canonical_fixture(root)
            class MutatingTokenizer(WordTokenizer):
                changed=False
                def __call__(self,text,**kwargs):
                    if not self.changed:
                        self.changed=True
                        with (corpus/'corpus.csv').open('a',encoding='utf8') as stream:stream.write('changed')
                    return super().__call__(text,**kwargs)
            spec={'model_name':'test_tokenizer','max_tokens':256,'hashes':{}}
            with patch('kazstyle.data.views.tokenizer_specs',return_value=[spec]),patch('kazstyle.data.views.load_dataset_tokenizer',return_value=(MutatingTokenizer(),spec)):
                with self.assertRaisesRegex(ValueError,'integrity'):
                    prepare([],reference,root/'out',corpus=corpus)
            self.assertFalse((root/'out').exists())

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


    def test_new_training_guard_rejects_reserved_material_before_tokenizer_load(self):
        from kazstyle.data.build_v3 import build
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'candidates.jsonl'
            source.write_text(json.dumps({'doc_id':'reserved','usage_status':'reserved_pending_review'})+'\n', encoding='utf8')
            with patch('kazstyle.data.build_v3.AutoTokenizer.from_pretrained') as loader:
                with self.assertRaisesRegex(ValueError, 'Reserved evaluation material'):
                    build(source, root/'out')
                loader.assert_not_called()
            self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
