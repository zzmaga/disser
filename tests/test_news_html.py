import unittest
from kazstyle.data.news_html import parse_article


class NewsHTMLTests(unittest.TestCase):
    def test_mixed_blockquote_keeps_quotation_and_removes_only_link_paragraph(self):
        raw = '''<div class="article-prose"><p>Қазақстанда жаңа кітапхана ашылды және көптеген қызықты кітап оқырманға ұсынылды.</p>
        <blockquote><p>Бұл <a href="/article/9-person">Сәтпаев</a>қа қолдау көрсетеді.</p>
        <p><strong><a href="/article/88-other">Басқа мақаланың тақырыбы</a></strong></p></blockquote>
        <p>Оқырмандар қуанды.</p></div>'''
        text, _, removed, _ = parse_article(raw, 'https://egemen.kz/article/1-current')
        self.assertNotIn('Басқа мақаланың тақырыбы', text)
        self.assertIn('Сәтпаевқа қолдау көрсетеді. Оқырмандар', text)
        self.assertNotIn('Сәтпаев қа', text)
        self.assertEqual(len(removed), 1)

    def test_standalone_recommendation_is_removed_but_inline_citation_retained(self):
        raw = '''<div class="article-prose"><p>Қазақстанда жаңа кітапхана ашылды және көптеген қызықты кітап оқырманға ұсынылды.</p>
        <p><a href="/article/88-other">Басқа мақаланың тақырыбы</a></p>
        <p>Бұл <a href="/article/9-source">зерттеу</a> нәтижесі.</p></div>'''
        text, _, _, _ = parse_article(raw, 'https://egemen.kz/article/1-current')
        self.assertNotIn('Басқа мақаланың тақырыбы', text)
        self.assertIn('Бұл зерттеу нәтижесі.', text)

    def test_nested_figure_does_not_crash_and_tables_are_flagged(self):
        raw = '''<div class="article-prose"><p>Қазақстанда жаңа кітапхана ашылды және көптеген қызықты кітап оқырманға ұсынылды.</p>
        <figure><figcaption>Фото түсіндірмесі</figcaption></figure><table><tr><td>Бір</td><td>Екі</td></tr></table></div>'''
        text, meta, _, _ = parse_article(raw, 'https://egemen.kz/article/1-current')
        self.assertNotIn('Фото түсіндірмесі', text)
        self.assertTrue(meta['contains_table'])


if __name__ == '__main__':
    unittest.main()
