import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import json
import requests

from kazstyle.data.refresh_news import parse_article, refresh


class RefreshNewsTests(unittest.TestCase):
    def test_failed_pages_are_preserved_for_review_and_checkpoint_resumes(self):
        calls = []
        class Response:
            status_code = 200
            text = 'User-agent: *\nAllow: /'
            def raise_for_status(self): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def iter_content(self, size):
                yield '<div class="article-prose"><p>Қазақстанда жаңа кітапхана ашылды және көптеген қызықты кітап оқырманға ұсынылды.</p></div>'.encode()
        class Session:
            def __init__(self): self.headers = {}
            def get(self, url, **kwargs): calls.append(url); return Response()
        with tempfile.TemporaryDirectory() as tmp, patch('kazstyle.data.refresh_news.requests.Session', Session), patch('kazstyle.data.refresh_news.time.sleep'):
            root = Path(tmp); source = root/'source.jsonl'; out = root/'out'
            rows = [{'doc_id':str(i), 'text':'Original text '+str(i), 'style':'publicistic',
                     'source_domain':'egemen.kz', 'source_url':f'https://egemen.kz/article/{i}'} for i in range(2)]
            source.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
            before = source.read_bytes(); refresh(source, out, limit=1, workers=1)
            result = [json.loads(l) for l in (out/'candidates.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(result[1]['source_extraction_status'], 'reextraction_failed_review_required')
            with self.assertRaises(FileExistsError): refresh(source, out)
            # Simulate an unfinished checkpoint in this synthetic temporary fixture.
            (out/'COMPLETE.json').unlink(); calls.clear()
            summary = refresh(source, out, limit=2, workers=1)
            self.assertEqual(summary['reextracted'], 2)
            self.assertNotIn('https://egemen.kz/article/0', calls)
            self.assertIn('https://egemen.kz/article/1', calls)
            self.assertEqual(source.read_bytes(), before)

    def test_related_link_removed_but_real_quote_and_inline_reference_survive(self):
        raw = '''<html><div class="article-prose">
        <p>Қазақстанда жаңа кітапхана ашылды және көптеген кітап оқырманға ұсынылды.</p>
        <blockquote><p><strong><a href="https://egemen.kz/article/99-other">Басқа жаңалық</a></strong></p></blockquote>
        <blockquote><p>Ғалым: «Кітап оқу адамға білім береді» деп айтты.</p></blockquote>
        <p>Осы <a href="/article/55-source">зерттеу</a> нәтижесі жарияланды.</p>
        <p>Суретті түсірген – Мәди ҒҰБАЙДУЛЛИН</p>
        </div><footer>Бұл сайттың төменгі бөлігі</footer></html>'''
        text, _, removed, _ = parse_article(raw, 'https://egemen.kz/article/1-current')
        self.assertNotIn('Басқа жаңалық', text)
        self.assertNotIn('ҒҰБАЙДУЛЛИН', text)
        self.assertNotIn('сайттың', text)
        self.assertIn('Кітап оқу адамға білім береді', text)
        self.assertIn('Осы зерттеу нәтижесі жарияланды.', text)
        self.assertEqual({r['kind'] for r in removed}, {'related_article_links', 'photo_credit_paragraph'})

    def test_quote_with_context_is_not_a_recommendation(self):
        raw = '''<div class="article-prose"><p>Қазақстанда жаңа кітапхана ашылды және оқырманға қызмет көрсетеді.</p>
        <blockquote><p>Автор <a href="/article/99-other">Басқа жаңалық</a> туралы пікір білдірді.</p></blockquote></div>'''
        text, _, removed, _ = parse_article(raw, 'https://egemen.kz/article/1-current')
        self.assertIn('Басқа жаңалық туралы пікір білдірді', text)
        self.assertEqual(removed, [])

    def test_layout_change_and_empty_article_fail_instead_of_collecting_navigation(self):
        for raw in ['<article>Сайт мәзірі</article>', '<div class="article-prose"></div>',
                    '<div class="article-prose">бір</div><div class="article-prose">екі</div>']:
            with self.assertRaises(ValueError): parse_article(raw, 'https://egemen.kz/article/1')


if __name__ == '__main__':
    unittest.main()
