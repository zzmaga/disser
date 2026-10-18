import json
from pathlib import Path
import tempfile
import unittest

from kazstyle.data.corpus import digest,file_hash,write_json,write_jsonl
from kazstyle.data.curation import prepare
from kazstyle.data.export_corpus import build


TEXT='Сәлем достар, кеше жаңа кітапханаға бардым, маған қатты ұнады және тағы барғым келеді.'


class CurationTests(unittest.TestCase):
    def fixture(self,root):
        rows=[]
        for doc_id,noun,genre,domain in [('a','кітапханаға','forum_answer',''),
                 ('b','мұражайға','forum_answer',''),('c','мектепке','customer_review','appstore')]:
            rows.append({'doc_id':doc_id,'text':TEXT.replace('кітапханаға',noun),
                         'style':'colloquial','genre':genre,'source_domain':'example.org',
                         'source_url':'https://example.org/'+doc_id,'parent_id':doc_id,
                         'upstream_domain':domain,'review_status':'unreviewed'})
        rows.append({**rows[0],'doc_id':'contact','text':TEXT+' 87011234567'})
        source=root/'source.jsonl';write_jsonl(source,rows)
        recipe=root/'recipe.json';write_json(recipe,{
            'purpose':'provisional_document_corpus','inputs':[
                {'path':str(source),'sha256':file_hash(source),'require_review':True}]})
        corpus=root/'corpus';build(recipe,corpus)
        return corpus,recipe,rows

    def packet(self,out):
        return list(map(json.loads,(out/'packet.jsonl').read_text(encoding='utf-8').splitlines()))

    def test_full_texts_strata_blinding_and_no_automatic_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);corpus,recipe,rows=self.fixture(root)
            out=root/'packet';result=prepare(corpus,recipe,out,per_stratum=1,max_words=40)
            shown=self.packet(out)
            self.assertEqual(result['documents'],2)
            self.assertEqual(result['labels_approved'],0)
            self.assertNotIn('contact',{r['doc_id'] for r in shown})
            original={r['doc_id']:r['text'] for r in rows}
            for row in shown:
                self.assertEqual(row['text'],original[row['doc_id']])
                self.assertEqual(row['text_sha256'],digest(row['text']))
                self.assertNotIn('source_url',row);self.assertNotIn('genre',row)
            other=root/'same';prepare(corpus,recipe,other,per_stratum=1,max_words=40)
            self.assertEqual((out/'packet.jsonl').read_bytes(),(other/'packet.jsonl').read_bytes())

    def test_previous_decisions_are_excluded_only_with_matching_text_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);corpus,recipe,rows=self.fixture(root)
            review=root/'review.json';data={'reviewer_kind':'assistant','decisions':[
                {'doc_id':'a','text_sha256':digest(rows[0]['text']),'keep':False}]}
            write_json(review,data)
            out=root/'packet';prepare(corpus,recipe,out,prior_reviews=[review])
            self.assertEqual({r['doc_id'] for r in self.packet(out)},{'b','c'})
            data['decisions'][0]['text_sha256']='stale';write_json(review,data)
            with self.assertRaisesRegex(ValueError,'Stale'):
                prepare(corpus,recipe,root/'stale',prior_reviews=[review])
            self.assertFalse((root/'stale').exists())

    def test_changed_recipe_or_corpus_cannot_generate_an_audit_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);corpus,recipe,_=self.fixture(root)
            changed=root/'other_recipe.json';changed.write_bytes(recipe.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'recipe differs'):
                prepare(corpus,changed,root/'wrong_recipe')
            (corpus/'audit.jsonl').write_text('{}\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'integrity'):
                prepare(corpus,recipe,root/'wrong_corpus')

    def test_invalid_limits_and_existing_outputs_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);corpus,recipe,_=self.fixture(root)
            for kwargs in [{'per_stratum':0},{'max_words':5,'min_words':8},{'per_stratum':True}]:
                with self.assertRaises(ValueError):prepare(corpus,recipe,root/'bad',**kwargs)
            out=root/'packet';prepare(corpus,recipe,out)
            with self.assertRaises(FileExistsError):prepare(corpus,recipe,out)


if __name__=='__main__':unittest.main()
