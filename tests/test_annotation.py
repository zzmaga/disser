import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from kazstyle.data.annotation import build_packet, compare_reviews, read_packet, validate_review, LABELS
from kazstyle.data.corpus import STYLES, digest, file_hash, write_json, write_jsonl
from kazstyle.data.journals import article_key, parse_abstract


class AnnotationTests(unittest.TestCase):
    def test_canonical_csv_audit_is_blind_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);corpus=root/'corpus';corpus.mkdir()
            rows=[{'text':f'Қазақ тіліндегі мәтін {i} толық көрсетілмейтін ұзын жалғасы бар',
                   'label':label} for i,label in enumerate(STYLES.values())]
            with (corpus/'corpus.csv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['text','label']);writer.writeheader();writer.writerows(rows)
            metadata=[{'doc_id':str(i),'corpus_row':i+1,'style':style,'text_sha256':digest(row['text']),
                       'source_domain':'private.example'} for i,(style,row) in enumerate(zip(STYLES,rows))]
            write_jsonl(corpus/'metadata.jsonl',metadata);write_json(corpus/'summary.json',{})
            (corpus/'by_style').mkdir()
            for style,row in zip(STYLES,rows):
                with (corpus/'by_style'/f'{style}.csv').open('w',encoding='utf-8',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['text','label']);writer.writeheader();writer.writerow(row)
            files=list(corpus.rglob('*.csv'))+[corpus/'metadata.jsonl',corpus/'summary.json']
            write_json(corpus/'COMPLETE.json',{p.relative_to(corpus).as_posix():file_hash(p) for p in files})
            packet=build_packet([],root/'packet',per_style=1,corpus=corpus,max_words=5)
            self.assertTrue(all(len(item['text'].split())==5 for item in packet['items']))
            self.assertNotIn('private.example',(root/'packet/reviewer_A.html').read_text(encoding='utf-8'))
            manifest=json.loads((root/'packet/manifest.json').read_text(encoding='utf-8'))
            self.assertFalse(manifest['external_test']);self.assertEqual(manifest['human_reviews_received'],0)
            self.assertIn(str(corpus/'COMPLETE.json'),manifest['inputs'])
            self.assertEqual(manifest['excerpt_count'],5)
            (corpus/'corpus.csv').write_text('tampered',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'integrity check failed'):
                build_packet([],root/'bad',per_style=1,corpus=corpus)
            self.assertFalse((root/'bad').exists())

    def test_audit_source_is_explicit_and_excerpt_limit_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for inputs,corpus,max_words in [([],None,350),([root/'a'],root/'b',350),([root/'a'],None,0)]:
                with self.subTest(inputs=inputs,corpus=corpus,max_words=max_words),self.assertRaises(ValueError):
                    build_packet(inputs,root/'out',corpus=corpus,max_words=max_words)
            self.assertFalse((root/'out').exists())

    def fixture(self, root):
        source=root/'candidates.jsonl'
        write_jsonl(source,[{'doc_id':str(i), 'style':style, 'text':f'Қазақ тіліндегі мысал {i} </script><script>danger()</script>',
                            'source_url':'https://secret.example/'+style,'source_domain':'secret.example',
                            'parent_id':str(i)} for i,style in enumerate(STYLES)])
        packet=build_packet([source],root/'packet',per_style=1)
        decisions=[{'id':r['id'],'text_sha256':r['text_sha256'],'label':'official','reason':''} for r in packet['items']]
        a={'schema_version':1,'packet_sha256':packet['packet_sha256'],'reviewer':'R1','independent_review':True,'decisions':decisions}
        return packet,a

    def test_blind_packet_escapes_html_and_excludes_source_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);packet,_=self.fixture(root)
            page=(root/'packet/reviewer_A.html').read_text(encoding='utf-8')
            self.assertNotIn('secret.example',page)
            self.assertNotIn('</script><script>danger()',page)
            self.assertNotIn('provisional_style',page)
            self.assertEqual(read_packet(root/'packet/packet.json'),packet)
            packet['items'][0]['text']='tampered'
            write_json(root/'packet/packet.json',packet)
            with self.assertRaisesRegex(ValueError,'hash mismatch'): read_packet(root/'packet/packet.json')

    def test_incomplete_stale_duplicate_and_wrong_text_reviews_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            packet,a=self.fixture(Path(directory))
            for mutation in ['incomplete','duplicate','stale','text','label','declaration','reason']:
                broken=copy.deepcopy(a)
                if mutation=='incomplete':broken['decisions'].pop()
                if mutation=='duplicate':broken['decisions'].append(broken['decisions'][0])
                if mutation=='stale':broken['packet_sha256']='wrong'
                if mutation=='text':broken['decisions'][0]['text_sha256']='wrong'
                if mutation=='label':broken['decisions'][0]['label']='unknown'
                if mutation=='declaration':broken['independent_review']=False
                if mutation=='reason':broken['decisions'][0]['label']='noise'
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate_review(packet,broken)

    def test_same_reviewer_rejected_and_disagreement_not_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);packet,a=self.fixture(root)
            b=copy.deepcopy(a);b['reviewer']=' r1 '
            write_json(root/'a.json',a);write_json(root/'b.json',b)
            with self.assertRaisesRegex(ValueError,'distinct'):
                compare_reviews(root/'packet/packet.json',root/'a.json',root/'b.json',root/'result')
            self.assertFalse((root/'result').exists())
            b['reviewer']='R2';b['decisions'][0]['label']='scientific';write_json(root/'b.json',b)
            result=compare_reviews(root/'packet/packet.json',root/'a.json',root/'b.json',root/'result')
            self.assertEqual(result['disagreements'],1)
            self.assertEqual(result['agreed_styles'],4)
            self.assertEqual(result['agreement'],.8)
            self.assertAlmostEqual(result['cohen_kappa_all_decisions'],0.)
            self.assertEqual(len((root/'result/needs_adjudication.jsonl').read_text(encoding='utf-8').splitlines()),1)

    def test_journal_extracts_body_not_navigation_and_deduplicates_locales(self):
        page='''<html lang="kk"><head><meta name="citation_title" content="Title"></head><body>
        <nav>Menu</nav><section class="item abstract"><h2 class="label">Аңдатпа</h2>
        <p>Зерттеу нәтижесі.</p><p>Түйін сөздер: тест</p></section><aside>Related news</aside>
        <a href="http://creativecommons.org/licenses/by-nc/4.0/">License</a></body></html>'''
        text,meta=parse_abstract(page)
        self.assertEqual(text.strip(),'Зерттеу нәтижесі.')
        self.assertEqual(meta['title'],'Title')
        self.assertTrue(meta['license_urls'])
        self.assertEqual(article_key('https://journal.example/index.php/j/kk/article/view/123'),
                         article_key('https://journal.example/index.php/j/en/article/view/123'))


if __name__=='__main__':unittest.main()
