"""Exercise the actual v6 profile through the same service used by the website."""
import json
import unittest
from pathlib import Path
from kazstyle.settings import artifact_path

PROFILE=Path('configs/deployments/expanded_v6_shared.json')


@unittest.skipUnless(PROFILE.exists(),'Requires the completed v6 deployment profile')
class V6InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kazstyle.inference.service import InferenceService
        cls.service=InferenceService(deployment=PROFILE)

    @classmethod
    def tearDownClass(cls):
        import gc
        del cls.service;gc.collect()

    def test_all_six_restore_saved_predictions_for_every_style(self):
        records={}
        for key,spec in self.service.models.items():
            name=spec['file'].replace('.joblib','_test_predictions.jsonl') if spec['kind']=='classical' else 'test_predictions.jsonl'
            records[key]={r['sample_id']:r for r in map(json.loads,(artifact_path(spec['run'])/name).read_text(encoding='utf-8').splitlines())}
        self.assertEqual(len(records),6)
        for label in range(5):
            sample=next(r for r in records['mbert'].values() if r['y_true']==label)
            response=self.service.classify(sample['text'],compare=True)
            self.assertEqual(response['excerpt'],sample['text'])
            for prediction in response['results']:
                self.assertEqual(prediction['style'],records[prediction['model']][sample['sample_id']]['predicted_name'])

    def test_url_changes_leave_all_predictions_unchanged(self):
        base='Қазақстандағы білім беру жүйесінде жаңа оқу бағдарламалары қабылданды.'
        a=self.service.classify(base+' https://adilet.zan.kz/doc',compare=True)
        b=self.service.classify(base+' https://ertegiler.kz/story',compare=True)
        self.assertEqual(a['excerpt'],base);self.assertEqual(a['excerpt'],b['excerpt'])
        self.assertEqual([r['style'] for r in a['results']],[r['style'] for r in b['results']])

    def test_long_text_uses_one_shared_limited_view(self):
        text=('Қазақ тіліндегі мәтіндерді зерттеу үшін тілдік деректер мен әдістер салыстырылады. '*80).strip()
        response=self.service.classify(text,compare=True)
        self.assertTrue(response['partial_text']);self.assertEqual(len(response['results']),6)
        for tokenizer,limit in self.service.constraints:
            self.assertLessEqual(len(tokenizer(response['excerpt'])['input_ids']),limit)


if __name__=='__main__':unittest.main()
