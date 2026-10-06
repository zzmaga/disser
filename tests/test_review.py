import json
import tempfile
import unittest
from pathlib import Path

from kazstyle.data.corpus import digest, file_hash, write_json, write_jsonl
from kazstyle.data.review import apply_review


class ReviewTests(unittest.TestCase):
    def test_all_style_review_requires_text_hash_and_preserves_reason(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); source = root/'input.jsonl'; review = root/'review.json'
            write_jsonl(source, [{'doc_id': 'a', 'style': 'scientific', 'text': 'example'}])
            record = {'schema_version': 2, 'scope': 'all_rows', 'reviewer_kind': 'assistant',
                      'reviewer': 'test_assistant', 'input_sha256': file_hash(source),
                      'decisions': [{'doc_id': 'a', 'keep': True, 'reason': 'provisional example', 'text_sha256': 'wrong', 'reviewed_extent': 'entire_candidate_text'}]}
            write_json(review, record)
            with self.assertRaisesRegex(ValueError, 'text hash mismatch'):
                apply_review(source, review, root/'out')
            record['decisions'][0]['text_sha256'] = digest('example')
            write_json(review, record); apply_review(source, review, root/'out')
            accepted = json.loads((root/'out/candidates.jsonl').read_text())
            self.assertEqual(accepted['review_decision']['reason'], 'provisional example')
            self.assertEqual(accepted['review_status'], 'assistant_reviewed_not_expert')
            self.assertEqual(accepted['usage_status'], 'review_pool_only')

    def test_stale_and_incomplete_review_cannot_approve_rows(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / 'input.jsonl'
            review = root / 'review.json'
            write_jsonl(source, [{'doc_id': 'a', 'style': 'colloquial', 'text': 'message'}])
            write_json(review, {'input_sha256': 'wrong', 'decisions': []})
            with self.assertRaisesRegex(ValueError, 'different candidate snapshot'):
                apply_review(source, review, root / 'out')
            write_json(review, {'input_sha256': file_hash(source), 'decisions': []})
            with self.assertRaisesRegex(ValueError, 'exactly one review decision'):
                apply_review(source, review, root / 'out')
            self.assertFalse((root / 'out').exists())
