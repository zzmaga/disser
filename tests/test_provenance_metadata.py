import json
import tempfile
import unittest
from pathlib import Path
from kazstyle.data.corpus import digest,file_hash,write_json,write_jsonl
from kazstyle.data.provenance import apply_prose_metadata
from tests.test_views import TEXT


class ProseAttributionTests(unittest.TestCase):
    def test_explicit_footer_removed_but_names_inside_story_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);source=root/'source.jsonl';plan=root/'plan.json'
            text='(Әңгіме) '+TEXT+' Автор Аты'
            write_jsonl(source,[{'doc_id':'a','text':text,'title':'Шығарма','parent_id':'publication','source_domain':'example.org'}])
            decision={'doc_id':'a','text_sha256':digest(text),'author_kind':'byline','author_name':'Автор Аты',
                      'evidence_field':'text','evidence':'Автор Аты','remove_suffix':'Автор Аты'}
            write_json(plan,{'input_sha256':file_hash(source),'decisions':[decision]})
            apply_prose_metadata(source,plan,root/'out')
            row=json.loads((root/'out/candidates.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(row['text'],TEXT)
            self.assertEqual(row['author_name'],'Автор Аты')
            self.assertEqual(row['work_group'],'publication')
            self.assertNotEqual(row['content_hash'],digest(text.casefold()))


if __name__=='__main__':unittest.main()
