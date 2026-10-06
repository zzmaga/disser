"""Local testing website: python manage.py serve, then http://127.0.0.1:8765."""
from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from kazstyle.settings import FRONTEND_DIR

from kazstyle.inference.service import InferenceService

WEB = FRONTEND_DIR
STATIC = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8')}
MAX_BODY_BYTES = 500_000


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
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
            self.wfile.write(body)

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
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > MAX_BODY_BYTES:
                    return self.reply(413, {'error': 'Пустой или слишком большой запрос.'})
                if self.headers.get_content_type() != 'application/json':
                    return self.reply(415, {'error': 'Ожидается JSON.'})
                payload = json.loads(self.rfile.read(length).decode('utf-8'))
                if not isinstance(payload, dict):
                    raise ValueError('Некорректный запрос.')
                if not isinstance(payload.get('compare', False), bool):
                    raise ValueError('Некорректный режим сравнения.')
                result = service.classify(payload.get('text'), payload.get('model'), payload.get('compare', False))
                return self.reply(200, result)
            except (ValueError, UnicodeDecodeError) as error:
                return self.reply(400, {'error': str(error)})
            except Exception:
                logging.exception('Local inference failed')
                return self.reply(500, {'error': 'Не удалось выполнить проверку. Подробности в журнале сервера.'})
    return Handler


def main():
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
