import unittest
import json
from pathlib import Path
import tempfile
from kazstyle.data.forum_replies import parse_replies, build


class ReplyExtractionTests(unittest.TestCase):
    def test_missing_answer_date_does_not_borrow_nested_comment_date(self):
        page = '''<div class="qa-a-list-item" id="a22">
        <div class="qa-a-item-content"><div itemprop="text">answer</div></div>
        <time itemprop="dateModified" datetime="2025-01-01"></time>
        <div class="qa-c-list-item" id="c23">
        <div class="qa-c-item-content"><div itemprop="text">comment</div></div>
        <time itemprop="dateCreated" datetime="2016-01-01"></time></div></div>'''
        answer, comment = list(parse_replies(page, 'https://surak.baribar.kz/21/'))
        self.assertIsNone(answer['published_at'])
        self.assertEqual(answer['modified_at'], '2025-01-01')
        self.assertEqual(comment['published_at'], '2016-01-01')
        self.assertIsNone(comment['modified_at'])

    def test_reserved_thread_cannot_be_reused_as_new_training_replies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'reserved.jsonl'
            source.write_text(json.dumps({'usage_status':'reserved_final_test',
                                          'source_url':'https://surak.baribar.kz/123/'})+'\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Reserved'):
                build([source], root/'out')
            self.assertFalse((root/'out').exists())

    def test_answer_does_not_swallow_nested_comment_or_question(self):
        page = '''<div class="qa-q-view-content"><div itemprop="text">question</div></div>
        <div class="qa-a-list-item" id="a22"><div class="qa-a-item-content"><div itemprop="text">answer</div></div>
        <span class="qa-a-item-who-data"><a class="qa-user-link" href="../user/one">one</a></span>
        <div class="qa-c-list-item" id="c23"><div class="qa-c-item-content"><div itemprop="text">comment</div></div>
        <span class="qa-c-item-who-data"><a class="qa-user-link" href="../user/two">two</a></span></div></div>'''
        rows = list(parse_replies(page, 'https://surak.baribar.kz/21/'))
        self.assertEqual([r['text'] for r in rows], ['answer', 'comment'])
        self.assertNotEqual(rows[0]['author_group'], rows[1]['author_group'])
        self.assertEqual(rows[1]['source_url'], 'https://surak.baribar.kz/21/#c23')

    def test_code_is_flagged_without_silently_rewriting_it(self):
        page = '<div class="qa-a-list-item" id="a12"><div class="qa-a-item-content"><div itemprop="text"><code>x=1</code> explanation</div></div></div>'
        row = next(parse_replies(page, 'https://surak.baribar.kz/11/'))
        self.assertTrue(row['contains_quote_or_code'])
        self.assertIn('x=1', row['text'])
        self.assertIsNone(row['author_group'])

    def test_reply_requires_a_real_post_identifier(self):
        self.assertEqual(list(parse_replies('<div class="qa-a-list-item"><div itemprop="text">text</div></div>', 'https://surak.baribar.kz/11/')), [])


if __name__ == '__main__':
    unittest.main()
