"""API contracts without model weights; HTTP framing is tested through Uvicorn."""
from http.client import HTTPConnection
import json
from pathlib import Path
import socket
import threading
import unittest

from fastapi.testclient import TestClient

from kazstyle.api.server import MAX_BODY_BYTES, create_app
from tests.http_support import running_server


class FakeService:
    default_model = 'test'
    dataset = Path('fixture')

    def available_models(self):
        return [{'id': 'test', 'name': 'Test'}]

    def available_styles(self):
        return ['Scientific']

    def classify(self, text, model, compare):
        if not text.strip():
            raise ValueError('Empty text')
        return {'text': text, 'model': model, 'compare': compare}


def client(service=None):
    return TestClient(create_app(service or FakeService()), base_url='http://127.0.0.1:8765')


class ApiContractTests(unittest.TestCase):
    def test_static_and_model_metadata(self):
        with client() as http:
            for path in ['/', '/app.js', '/style.css', '/api/models', '/api/health']:
                response = http.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
                self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
                self.assertTrue(response.content)
            self.assertEqual(http.get('/api/models').json()['default'], 'test')
            self.assertEqual(http.get('/api/health').json(), {'status': 'ready'})

    def test_openapi_describes_the_typed_request(self):
        with client() as http:
            schema = http.get('/openapi.json').json()
            self.assertIn('/api/classify', schema['paths'])
            fields = schema['components']['schemas']['ClassifyRequest']
            self.assertEqual(fields['properties']['compare']['type'], 'boolean')
            self.assertEqual(fields['properties']['text']['maxLength'], 100_000)
            self.assertFalse(fields['additionalProperties'])

    def test_rejects_invalid_payloads_and_types_without_echoing_input(self):
        with client() as http:
            bodies = [b'{', b'\xff', b'[]', b'{}', b'{"text":null}', b'{"text":123}',
                      b'{"text":"secret-input","compare":"yes"}', b'{"text":"ok","model":5}',
                      b'{"text":"ok","unexpected":true}']
            for body in bodies:
                with self.subTest(body=body):
                    response = http.post('/api/classify', content=body, headers={'Content-Type':'application/json'})
                    self.assertEqual(response.status_code, 400)
                    self.assertIn('error', response.json())
                    self.assertNotIn('secret-input', response.text)
            result = http.post('/api/classify', json={'text':'Қазақша мәтін', 'compare':True})
            self.assertEqual(result.status_code, 200)
            self.assertTrue(result.json()['compare'])
            self.assertIsNone(result.json()['model'])

    def test_text_limit_and_empty_text(self):
        with client() as http:
            for text in ['', '   ', 'a'*100_001]:
                self.assertEqual(http.post('/api/classify', json={'text':text}).status_code, 400)
            self.assertEqual(http.post('/api/classify', json={'text':'a'*100_000}).status_code, 200)

    def test_limits_type_origin_and_route(self):
        with client() as http:
            cases = [({'Content-Type':'text/plain'}, 415),
                     ({'Content-Type':'application/json', 'Content-Length':str(MAX_BODY_BYTES+1)}, 413),
                     ({'Content-Type':'application/json', 'Origin':'https://example.org'}, 403),
                     ({'Content-Type':'application/json', 'Transfer-Encoding':'chunked'}, 400),
                     ({'Content-Type':'application/json', 'Content-Length':'invalid'}, 400)]
            for headers, expected in cases:
                with self.subTest(headers=headers):
                    result = http.post('/api/classify', content=b'{}', headers=headers)
                    self.assertEqual(result.status_code, expected)
                    self.assertIn('error', result.json())
            self.assertEqual(http.post('/api/unknown', json={}).status_code, 404)
            self.assertEqual(http.get('/missing-file').status_code, 404)
            self.assertEqual(http.get('/../configs/deployment.json').status_code, 404)

    def test_local_browser_origins_and_default_comparison(self):
        with client() as http:
            for origin in ['http://127.0.0.1:8765', 'http://localhost:8765']:
                result = http.post('/api/classify', json={'text':'Мәтін'}, headers={'Origin':origin})
                self.assertEqual(result.status_code, 200)
                self.assertFalse(result.json()['compare'])
            self.assertEqual(http.post('/api/classify', json={'text':'Мәтін'},
                                       headers={'Origin':'http://localhost:9999'}).status_code, 403)

    def test_busy_request_is_rejected_and_slot_released_after_error(self):
        entered, release = threading.Event(), threading.Event()

        class BlockingService(FakeService):
            def classify(self, text, model, compare):
                if text == 'block':
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError('Test synchronization timeout')
                    raise ValueError('Expected validation failure')
                return super().classify(text, model, compare)

        with client(BlockingService()) as http:
            responses = []
            thread = threading.Thread(target=lambda: responses.append(http.post('/api/classify', json={'text':'block'})))
            thread.start()
            try:
                self.assertTrue(entered.wait(3))
                self.assertEqual(http.post('/api/classify', json={'text':'other'}).status_code, 503)
                self.assertEqual(http.get('/api/health').status_code, 200)
            finally:
                release.set()
                thread.join(timeout=5)
            self.assertEqual(responses[0].status_code, 400)
            self.assertEqual(http.post('/api/classify', json={'text':'other'}).status_code, 200)

    def test_inference_failure_returns_safe_error_and_releases_slot(self):
        class FailingService(FakeService):
            def classify(self, text, model, compare):
                if text == 'failure':
                    raise RuntimeError('internal-details')
                return super().classify(text, model, compare)

        with client(FailingService()) as http:
            with self.assertLogs(level='ERROR'):
                result = http.post('/api/classify', json={'text':'failure'})
            self.assertEqual(result.status_code, 500)
            self.assertNotIn('internal-details', result.text)
            self.assertEqual(http.post('/api/classify', json={'text':'next'}).status_code, 200)

    def test_uvicorn_handles_http_framing_and_a_real_request(self):
        with running_server(create_app(FakeService(), request_timeout=.15)) as address:
            for length in [b'Content-Length: 2\r\nContent-Length: 3', b'Content-Length: nope']:
                with socket.create_connection(address, timeout=3) as conn:
                    conn.sendall(b'POST /api/classify HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n'+length+b'\r\n\r\n{}')
                    self.assertIn(b' 400 ', conn.recv(4096).split(b'\r\n',1)[0])
            with socket.create_connection(address, timeout=3) as conn:
                conn.sendall(b'POST /api/classify HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{')
                self.assertIn(b' 408 ', conn.recv(4096).split(b'\r\n',1)[0])
            connection = HTTPConnection(*address, timeout=3)
            try:
                connection.request('POST', '/api/classify', json.dumps({'text':'Kazakh text'}),
                                   {'Content-Type':'application/json'})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())['text'], 'Kazakh text')
            finally:
                connection.close()


if __name__ == '__main__':
    unittest.main()
