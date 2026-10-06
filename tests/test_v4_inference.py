"""Real six-model deployment: saved predictions, URL invariance and shared limits."""
import json
import unittest
from pathlib import Path
from kazstyle.settings import artifact_path


PROFILE=Path('configs/deployments/text_only_v4_shared.json')


@unittest.skipUnless(PROFILE.exists(),'Requires the completed v4 deployment profile')
class V4InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kazstyle.inference.service import InferenceService
        cls.service=InferenceService(deployment=PROFILE)

    @classmethod
    def tearDownClass(cls):
        import gc
        del cls.service
        gc.collect()

    def test_six_models_reproduce_saved_prediction_for_each_style(self):
        records={}
        for key,spec in self.service.models.items():
            path=artifact_path(spec['run'])/(spec['file'].replace('.joblib','_test_predictions.jsonl') if spec['kind']=='classical' else 'test_predictions.jsonl')
            records[key]={row['sample_id']:row for row in map(json.loads,path.read_text(encoding='utf-8').splitlines())}
        self.assertEqual(len(records),6)
        for label in range(5):
            sample=next(row for row in records['mbert'].values() if row['y_true']==label)
            actual=self.service.classify(sample['text'],compare=True)
            self.assertEqual(actual['excerpt'],sample['text'])
            for result in actual['results']:
                self.assertEqual(result['style'],records[result['model']][sample['sample_id']]['predicted_name'])

    def test_switching_url_keeps_input_and_all_six_predictions(self):
        text='Қазақстандағы білім беру жүйесінде жаңа оқу бағдарламалары қабылданды.'
        a=self.service.classify(text+' https://adilet.zan.kz/document/1',compare=True)
        b=self.service.classify(text+' https://ertegiler.kz/story/123',compare=True)
        self.assertEqual(a['excerpt'],text);self.assertEqual(a['excerpt'],b['excerpt'])
        self.assertEqual([r['style'] for r in a['results']],[r['style'] for r in b['results']])

    def test_long_text_respects_both_tokenizers_with_identical_excerpt(self):
        from kazstyle.data.corpus import load_manifest
        source,_=load_manifest(Path('data/processed/text_only_v3'))
        changed=json.loads(Path('data/processed/text_only_v4_shared/derivation.json').read_text(encoding='utf-8'))['changes'][0]
        sample=source[source.doc_id==changed['doc_id']].iloc[0]
        result=self.service.classify(sample.text,compare=True)
        for tokenizer,limit in self.service.constraints:
            self.assertLessEqual(len(tokenizer(result['excerpt'])['input_ids']),limit)
        self.assertTrue(result['partial_text'])
        self.assertEqual(len(result['results']),6)


if __name__=='__main__':unittest.main()
