"""Local testing website: python manage.py serve, then http://127.0.0.1:8765."""
from __future__ import annotations

import argparse
import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from kazstyle.settings import FRONTEND_DIR

WEB = FRONTEND_DIR
STATIC = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8')}
MAX_BODY_BYTES = 500_000
REQUEST_TIMEOUT_SECONDS = 10


def handler_for(service):
    # Admit one inference request per service. The UI can retry instead of
    # accumulating threads waiting for the same model lock and CPU memory.
    inference_slot = threading.BoundedSemaphore(1)

    class Handler(BaseHTTPRequestHandler):
        timeout = REQUEST_TIMEOUT_SECONDS

        def log_message(self, format, *args):
            pass  # Do not save pasted text, request paths or browser traffic.

        def reply(self, status, data, content_type='application/json; charset=utf-8'):
            body = json.dumps(data, ensure_ascii=False).encode('utf-8') if isinstance(data, dict) else data
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
                self.wfile.flush()
                # HTTP/1.0 closes after this response. Send FIN before close so
                # Windows clients receive early rejections even with unread input.
                self.connection.shutdown(socket.SHUT_WR)
                if self.command == 'POST' and not getattr(self, '_body_consumed', False):
                    # Bounded lingering close: a late request-body packet must not
                    # reset the connection and discard our 4xx response on Windows.
                    deadline = time.monotonic() + .2
                    remaining = MAX_BODY_BYTES
                    while remaining and time.monotonic() < deadline:
                        self.connection.settimeout(max(.001, deadline-time.monotonic()))
                        chunk = self.connection.recv(min(65536, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
            except OSError:
                pass

        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path == '/api/health':
                return self.reply(200, {'status': 'ready'})
            if path == '/api/models':
                return self.reply(200, {'models': service.available_models(), 'default': service.default_model,
                                       'styles':service.available_styles(),'dataset':service.dataset.name})
            if path in STATIC:
                filename, kind = STATIC[path]
                return self.reply(200, (WEB / filename).read_bytes(), kind)
            self.reply(404, {'error': 'Страница не найдена.'})

        def do_POST(self):
            if self.path != '/api/classify':
                return self.reply(404, {'error': 'Маршрут не найден.'})
            # The browser must call this loopback API from this site's own origin.
            origin = self.headers.get('Origin')
            port = self.server.server_port
            if origin and origin not in {f'http://127.0.0.1:{port}', f'http://localhost:{port}'}:
                return self.reply(403, {'error': 'Запрос разрешён только с локального сайта.'})
            try:
                if self.headers.get('Transfer-Encoding'):
                    raise ValueError('Потоковая передача запроса не поддерживается.')
                lengths = self.headers.get_all('Content-Length', [])
                if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                    raise ValueError('Некорректная длина запроса.')
                length = int(lengths[0])
                if length <= 0 or length > MAX_BODY_BYTES:
                    return self.reply(413, {'error': 'Пустой или слишком большой запрос.'})
                if self.headers.get_content_type() != 'application/json':
                    return self.reply(415, {'error': 'Ожидается JSON.'})
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError('Запрос передан не полностью.')
                self._body_consumed = True
                payload = json.loads(body.decode('utf-8'))
                if not isinstance(payload, dict):
                    raise ValueError('Некорректный запрос.')
                if not isinstance(payload.get('compare', False), bool):
                    raise ValueError('Некорректный режим сравнения.')
                if not inference_slot.acquire(blocking=False):
                    return self.reply(503, {'error': 'Модель занята другим запросом. Повторите проверку позже.'})
                try:
                    result = service.classify(payload.get('text'), payload.get('model'), payload.get('compare', False))
                finally:
                    inference_slot.release()
                return self.reply(200, result)
            except TimeoutError:
                return self.reply(408, {'error': 'Истекло время передачи запроса.'})
            except (BrokenPipeError, ConnectionResetError):
                # Closing a browser tab is not an inference failure.
                return
            except (ValueError, UnicodeDecodeError) as error:
                return self.reply(400, {'error': str(error)})
            except Exception:
                logging.exception('Local inference failed')
                return self.reply(500, {'error': 'Не удалось выполнить проверку. Подробности в журнале сервера.'})
    return Handler


def main():
    from kazstyle.inference.service import InferenceService

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--deployment', help='Optional model registry JSON')
    args = parser.parse_args()
    service = InferenceService(deployment=args.deployment)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(service))
    print(f'Website ready: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
