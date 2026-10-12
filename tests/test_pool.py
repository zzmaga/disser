import tempfile
import unittest
from pathlib import Path

from kazstyle.data.corpus import digest,file_hash,write_json,write_jsonl
from kazstyle.data.pool import merge
from kazstyle.data.sampling import diverse_sample
import pandas as pd


class SamplingTests(unittest.TestCase):
    def test_missing_metadata_does_not_drop_documents_and_empty_keeps_schema(self):
        frame = pd.DataFrame([{'doc_id':str(i), 'source_domain':domain, 'genre':genre}
                              for i, (domain, genre) in enumerate([
                                  ('a', 'news'), ('a', 'news'), ('b', 'news'), (None, None)])])
        result = diverse_sample(frame, 20, 42)
        self.assertEqual(set(result.doc_id), set(frame.doc_id))
        self.assertEqual(len(result), len(frame))
        self.assertEqual(list(diverse_sample(frame, 0, 42).columns), list(frame.columns))
        self.assertEqual(diverse_sample(frame, 2, 42).source_domain.nunique(), 2)
        self.assertEqual(diverse_sample(frame, 3, 42).doc_id.tolist(), diverse_sample(frame, 3, 42).doc_id.tolist())


class DevelopmentPoolTests(unittest.TestCase):
    def test_exclusions_are_hash_bound_and_reserved_sources_blocked(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); source=root/'source.jsonl';recipe=root/'recipe.json'
            row={'doc_id':'a','text':'example','style':'literary','source_domain':'example.org','genre':'story'}
            write_jsonl(source,[row])
            plan={'purpose':'provisional_development_pool','inputs':[{'path':str(source),'sha256':file_hash(source)}],
                  'max_per_style':40,'seed':42,'limitations':['test fixture'],
                  'exclusions':[{'doc_id':'a','text_sha256':'stale','reason':'test'}]}
            write_json(recipe,plan)
            with self.assertRaisesRegex(ValueError,'matching text hash'):
                merge(recipe,root/'out')
            plan['exclusions']=[];write_jsonl(source,[{**row,'usage_status':'reserved_source'}])
            plan['inputs'][0]['sha256']=file_hash(source);write_json(recipe,plan)
            with self.assertRaisesRegex(ValueError,'Reserved'):
                merge(recipe,root/'out')
            self.assertFalse((root/'out').exists())

    def test_conflicting_versions_cannot_be_silently_merged(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);source=root/'source.jsonl';recipe=root/'recipe.json'
            write_jsonl(source,[{'doc_id':'a','text':'first','style':'literary'},
                               {'doc_id':'a','text':'changed','style':'literary'}])
            write_json(recipe,{'purpose':'provisional_development_pool','inputs':[{'path':str(source),'sha256':file_hash(source)}]})
            with self.assertRaisesRegex(ValueError,'Conflicting document versions'):
                merge(recipe,root/'out')


if __name__=='__main__':unittest.main()
