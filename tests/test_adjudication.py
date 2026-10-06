import copy
import json
import tempfile
import unittest
from pathlib import Path

from kazstyle.data.annotation import build_packet,compare_reviews
from kazstyle.data.adjudication import finalize_annotations
from kazstyle.data.corpus import STYLES,write_json,write_jsonl


class AdjudicationTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'source.jsonl'
        rows=[{'doc_id':str(i),'style':style,'text':('мысал0 '*400).strip() if i==0 else f'Қазақша шағын мысал {i}',
               'source_url':'https://example.org/'+str(i),'source_domain':'example.org','parent_id':'parent-'+str(i)}
              for i,style in enumerate(STYLES)]
        write_jsonl(source,rows);packet=build_packet([source],root/'packet',per_style=1)
        a={'schema_version':1,'packet_sha256':packet['packet_sha256'],'reviewer':'R1','independent_review':True,
           'decisions':[{'id':r['id'],'text_sha256':r['text_sha256'],'label':'official','reason':''} for r in packet['items']]}
        b=copy.deepcopy(a);b['reviewer']='R2';b['decisions'][0]['label']='scientific'
        write_json(root/'a.json',a);write_json(root/'b.json',b)
        compare_reviews(root/'packet/packet.json',root/'a.json',root/'b.json',root/'comparison')
        template=json.loads((root/'comparison/adjudication_template.json').read_text(encoding='utf-8'))
        template.update(adjudicator='R3',confirmed_read_shown_texts=True)
        template['decisions'][0].update(label='scientific',reason='Synthetic test adjudication, not a real corpus review')
        write_json(root/'adjudication.json',template)
        return template

    def test_export_only_shown_excerpt_and_keep_original_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            result=finalize_annotations(root/'packet/packet.json',root/'a.json',root/'b.json',root/'adjudication.json',root/'export')
            self.assertEqual(result['accepted'],5);self.assertEqual(result['resolved_disagreements'],1)
            rows=list(map(json.loads,(root/'export/reviewed_candidates.jsonl').read_text(encoding='utf-8').splitlines()))
            long=next(r for r in rows if r['source_document_id']=='0')
            self.assertEqual(len(long['text'].split()),350)
            self.assertEqual(long['parent_id'],'parent-0')
            self.assertNotEqual(long['doc_id'],long['source_document_id'])
            self.assertFalse(result['training_modified'])

    def test_unresolved_stale_or_unexplained_adjudication_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);good=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'Unresolved'):
                finalize_annotations(root/'packet/packet.json',root/'a.json',root/'b.json',None,root/'export')
            for mutation in ['pair','reason','incomplete','text','declaration']:
                bad=copy.deepcopy(good)
                if mutation=='pair':bad['review_hashes'][0]='wrong'
                if mutation=='reason':bad['decisions'][0]['reason']=''
                if mutation=='incomplete':bad['decisions']=[]
                if mutation=='text':bad['decisions'][0]['text_sha256']='wrong'
                if mutation=='declaration':bad['confirmed_read_shown_texts']=False
                write_json(root/'bad.json',bad)
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    finalize_annotations(root/'packet/packet.json',root/'a.json',root/'b.json',root/'bad.json',root/'export')
                self.assertFalse((root/'export').exists())

    def test_changed_source_snapshot_cannot_acquire_verified_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            with (root/'source.jsonl').open('a',encoding='utf-8') as stream:stream.write('\n')
            with self.assertRaisesRegex(ValueError,'snapshot changed'):
                finalize_annotations(root/'packet/packet.json',root/'a.json',root/'b.json',root/'adjudication.json',root/'export')
            self.assertFalse((root/'export').exists())


if __name__=='__main__':unittest.main()
