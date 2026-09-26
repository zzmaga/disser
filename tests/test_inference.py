"""Integration checks against actual saved predictions; no network or retraining."""
import json
import unittest
from pathlib import Path


@unittest.skipUnless(Path('artifacts/pilot_v2_roberta_concat4/results.json').exists(), 'Requires trained pilot artifacts')
class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kazstyle.inference.service import InferenceService
        cls.service = InferenceService()

    def test_all_models_reproduce_saved_predictions(self):
        paths = {
            'roberta': 'pilot_v2_roberta_last/test_predictions.jsonl',
            'roberta_concat4': 'pilot_v2_roberta_concat4/test_predictions.jsonl',
            'logreg': 'pilot_v2_classical/word_tfidf_logreg_test_predictions.jsonl',
            'word_svm': 'pilot_v2_classical/word_tfidf_linear_svc_test_predictions.jsonl',
            'char_svm': 'pilot_v2_classical/char_tfidf_linear_svc_test_predictions.jsonl',
        }
        records = {key: [json.loads(line) for line in (Path('artifacts') / path).read_text(encoding='utf-8').splitlines()]
                   for key, path in paths.items()}
        for label in range(3):
            sample = next(r for r in records['roberta'] if r['y_true'] == label)
            result = self.service.classify(sample['text'], compare=True)
            self.assertEqual(result['excerpt'], sample['text'])
            for prediction in result['results']:
                expected = next(r for r in records[prediction['model']] if r['sample_id'] == sample['sample_id'])
                self.assertEqual(prediction['style'], expected['predicted_name'])

    def test_short_input_is_allowed_with_warning(self):
        result = self.service.classify('Сәлем, бүгін қалайсың?', model_key='logreg')
        self.assertEqual(len(result['results']), 1)
        self.assertTrue(result['warnings'])

    def test_invalid_input_is_rejected(self):
        for value in ['', '  ', None, 123, '123 !!!', 'https://example.com']:
            with self.assertRaises(ValueError):
                self.service.classify(value)
        with self.assertRaises(ValueError):
            self.service.classify('Қазақша мәтін', model_key='missing')


if __name__ == '__main__':
    unittest.main()
