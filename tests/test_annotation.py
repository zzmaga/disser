import copy
import json
import tempfile
import unittest
from pathlib import Path

from kazstyle.data.annotation import build_packet, compare_reviews, read_packet, validate_review, LABELS
from kazstyle.data.corpus import STYLES, write_json, write_jsonl
from kazstyle.data.journals import article_key, parse_abstract


class AnnotationTests(unittest.TestCase):
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
