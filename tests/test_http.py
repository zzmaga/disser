"""HTTP regression: real static files and a real local classifier after moving modules."""
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from kazstyle.settings import PROJECT_ROOT, artifact_path


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
        from kazstyle.api.server import handler_for
        from kazstyle.inference.service import InferenceService
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(InferenceService()))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            for path in ['/', '/app.js', '/style.css', '/api/health', '/api/models']:
                with urlopen(base + path, timeout=10) as response:
                    self.assertEqual(response.status, 200)
                    self.assertTrue(response.read())
            payload = json.dumps({'text': 'Бүгін білім мен ғылым туралы хабар жарияланды.', 'model': 'logreg'}).encode()
            request = Request(base+'/api/classify', data=payload, headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=30) as response:
                result = json.load(response)
            self.assertEqual(result['results'][0]['model'], 'logreg')
            self.assertIn(result['results'][0]['style'], ['Formal', 'Publicist', 'Artistic', 'Scientific', 'Colloquial'])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
