"""Synthetic API contract checks, no corpus or model loading."""
import unittest

from fastapi.testclient import TestClient
from kazstyle.api.preparation import app


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_and_openapi(self):
        self.assertFalse(self.client.get('/health').json()['model_loaded'])
        self.assertIn('/api/prepare', self.client.get('/openapi.json').json()['paths'])

    def test_real_cleaner_preserves_kazakh_and_deduplicates(self):
        response = self.client.post('/api/prepare', json={'records': [
            {'id': 'a', 'text': '<p>Қазақ тілі. 04.10.2026</p> https://example.org'},
            {'id': 'b', 'text': 'Қазақ тілі. 04.10.2026'},
            {'id': 'c', 'text': 'https://example.net'},
        ]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['records'][0]['text'], 'Қазақ тілі. 04.10.2026')
        self.assertEqual(response.json()['summary']['kept_rows'], 1)
        self.assertEqual(response.json()['summary']['excluded_rows'], 2)

    def test_bad_input_and_duplicate_ids_rejected(self):
        for records in [[], [{'id': 'a', 'text': 12}], [{'id': ' ', 'text': 'hello'}],
                        [{'id': 'a', 'text': 'x', 'label': 0}],
                        [{'id': 'a', 'text': 'one'}, {'id': 'a', 'text': 'two'}],
                        [{'id': str(i), 'text': 'word'} for i in range(21)],
                        [{'id': 'a', 'text': 'x' * 10001}]]:
            with self.subTest(records=records[:1]):
                self.assertEqual(self.client.post('/api/prepare', json={'records': records}).status_code, 422)

    def test_url_changes_do_not_change_cleaned_text(self):
        def call(url):
            return self.client.post('/api/prepare', json={'records': [
                {'id': 'x', 'text': 'Ғылым және білім. '+url}]}).json()['records']
        self.assertEqual(call('https://example.org'), call('https://other.net'))
