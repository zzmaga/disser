"""Exercise restored weights and invariance to irrelevant URL changes."""
import json
import unittest
from pathlib import Path
from kazstyle.settings import artifact_path


@unittest.skipUnless((artifact_path('text_only_v3_roberta_concat4')/'results.json').exists(),
                     'Requires completed v3 models')
class V3InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kazstyle.inference.service import InferenceService
        cls.service = InferenceService(deployment=Path('configs/deployments/text_only_v3.json'))

    @classmethod
    def tearDownClass(cls):
        import gc
        del cls.service
        gc.collect()

    def test_all_five_models_reproduce_fixed_predictions_for_each_class(self):
        records = {}
        for key, spec in self.service.models.items():
            name = spec['file'].replace('.joblib', '_test_predictions.jsonl') if spec['kind'] == 'classical' else 'test_predictions.jsonl'
            records[key] = [json.loads(line) for line in (artifact_path(spec['run']) / name).read_text(encoding='utf-8').splitlines()]
        for label in range(5):
            sample = next(row for row in records['roberta'] if row['y_true'] == label)
            actual = self.service.classify(sample['text'], compare=True)
            self.assertEqual(actual['excerpt'], sample['text'])
            for prediction in actual['results']:
                expected = next(row for row in records[prediction['model']] if row['sample_id'] == sample['sample_id'])
                self.assertEqual(prediction['style'], expected['predicted_name'])

    def test_switching_publisher_url_does_not_change_any_prediction(self):
        text = 'Қазақстандағы білім беру жүйесінде жаңа оқу бағдарламалары қабылданды.'
        first = self.service.classify(text + ' https://adilet.zan.kz/document/1', compare=True)
        second = self.service.classify(text + ' https://ertegiler.kz/story/123', compare=True)
        self.assertEqual(first['excerpt'], text)
        self.assertEqual(first['excerpt'], second['excerpt'])
        self.assertEqual([r['style'] for r in first['results']], [r['style'] for r in second['results']])
