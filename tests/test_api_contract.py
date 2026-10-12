"""HTTP behavior without model weights, datasets, GPU or external requests."""
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading
import unittest

from kazstyle.api.server import MAX_BODY_BYTES, handler_for


class FakeService:
    default_model = 'test'
    dataset = Path('fixture')

    def available_models(self):
        return [{'id': 'test', 'name': 'Test'}]

    def available_styles(self):
        return ['Scientific']

    def classify(self, text, model, compare):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Empty text')
        return {'text': text, 'compare': compare}


@contextmanager
def running_server(service=None, timeout=1):
    handler = handler_for(service or FakeService())
    handler.timeout = timeout
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def request(server, body, headers=None, path='/api/classify'):
    conn = HTTPConnection('127.0.0.1', server.server_port, timeout=3)
    try:
        conn.request('POST', path, body, headers or {'Content-Type': 'application/json'})
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


class ApiContractTests(unittest.TestCase):
    def test_static_and_model_metadata(self):
        with running_server() as server:
            for path in ['/', '/app.js', '/style.css', '/api/models', '/api/health']:
                conn = HTTPConnection('127.0.0.1', server.server_port, timeout=3)
                try:
                    conn.request('GET', path)
                    response = conn.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.getheader('Cache-Control'), 'no-store')
                    self.assertTrue(response.read())
                finally:
                    conn.close()

    def test_rejects_invalid_payloads_and_types(self):
        with running_server() as server:
            for body in [b'{', b'\xff', b'[]', b'{"text":"ok","compare":"yes"}', b'{}']:
                self.assertEqual(request(server, body)[0], 400)
            status, result = request(server, json.dumps({'text': 'Kazakh text', 'compare': True}))
            self.assertEqual(status, 200)
            self.assertTrue(result['compare'])

    def test_limits_type_origin_and_route(self):
        with running_server() as server:
            cases = [({'Content-Type': 'text/plain'}, 415),
                     ({'Content-Type': 'application/json', 'Content-Length': str(MAX_BODY_BYTES+1)}, 413),
                     ({'Content-Type': 'application/json', 'Origin': 'https://example.org'}, 403),
                     ({'Content-Type': 'application/json', 'Transfer-Encoding': 'chunked'}, 400)]
            for headers, expected in cases:
                with self.subTest(headers=headers):
                    self.assertEqual(request(server, b'{}', headers)[0], expected)
            self.assertEqual(request(server, b'{}', path='/api/unknown')[0], 404)

    def test_duplicate_or_invalid_length_and_truncated_body(self):
        with running_server() as server:
            for length in [b'Content-Length: 2\r\nContent-Length: 2',
                           b'Content-Length: nope', b'Content-Length: 100']:
                with socket.create_connection(server.server_address, timeout=3) as conn:
                    conn.sendall(b'POST /api/classify HTTP/1.0\r\nContent-Type: application/json\r\n'+length+b'\r\n\r\n{}')
                    conn.shutdown(socket.SHUT_WR)
                    self.assertIn(b' 400 ', conn.recv(4096).split(b'\r\n', 1)[0])

    def test_incomplete_body_times_out(self):
        with running_server(timeout=.15) as server:
            with socket.create_connection(server.server_address, timeout=3) as conn:
                conn.sendall(b'POST /api/classify HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{')
                self.assertIn(b' 408 ', conn.recv(4096).split(b'\r\n', 1)[0])

    def test_busy_request_is_rejected_and_slot_released_after_error(self):
        entered, release = threading.Event(), threading.Event()

        class BlockingService(FakeService):
            def classify(self, text, model, compare):
                if text == 'block':
                    entered.set()
                    if not release.wait(3):
                        raise RuntimeError('Test synchronization timeout')
                    raise ValueError('Expected validation failure')
                return super().classify(text, model, compare)

        with running_server(BlockingService()) as server:
            responses = []
            thread = threading.Thread(target=lambda: responses.append(request(server, '{"text":"block"}')))
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(request(server, '{"text":"other"}')[0], 503)
            finally:
                release.set()
                thread.join(timeout=3)
            self.assertEqual(responses[0][0], 400)
            self.assertEqual(request(server, '{"text":"other"}')[0], 200)


if __name__ == '__main__':
    unittest.main()
