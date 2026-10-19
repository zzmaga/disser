"""Real local classifier served over the same Uvicorn stack as the website."""
import json
import unittest
from urllib.request import Request, urlopen

from kazstyle.settings import PROJECT_ROOT, artifact_path
from tests.http_support import running_server


def active_local_classifier_available():
    profile = PROJECT_ROOT/'configs/deployment.json'
    if not profile.exists():
        return False
    config = json.loads(profile.read_text(encoding='utf8'))
    model = config['models']['logreg']
    return ((PROJECT_ROOT/config['dataset']/'config.json').exists()
            and (artifact_path(model['run'])/model['file']).exists())


@unittest.skipUnless(active_local_classifier_available(), 'Requires the active local deployment and classifier')
class HttpTests(unittest.TestCase):
    def test_frontend_and_local_prediction(self):
        from kazstyle.api.server import create_app
        from kazstyle.inference.service import InferenceService
        service = InferenceService()
        text = 'Бүгін білім мен ғылым туралы хабар жарияланды.'
        expected = service.classify(text, 'logreg')['results'][0]['style']
        with running_server(create_app(service)) as address:
            base = f'http://{address[0]}:{address[1]}'
            for path in ['/', '/app.js', '/style.css', '/api/health', '/api/models', '/openapi.json']:
                with urlopen(base+path, timeout=10) as response:
                    self.assertEqual(response.status, 200)
                    self.assertTrue(response.read())
            request = Request(base+'/api/classify', data=json.dumps({'text':text, 'model':'logreg'}).encode(),
                              headers={'Content-Type':'application/json'})
            with urlopen(request, timeout=30) as response:
                result = json.load(response)
            self.assertEqual(result['results'][0]['model'], 'logreg')
            self.assertEqual(result['results'][0]['style'], expected)
