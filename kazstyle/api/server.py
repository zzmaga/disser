"""FastAPI application for the local website; run with ``manage.py serve``."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import logging
import threading

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect
import uvicorn

from kazstyle.settings import FRONTEND_DIR

MAX_BODY_BYTES = 500_000
REQUEST_TIMEOUT_SECONDS = 10
HEADERS = {
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
    'Content-Security-Policy': "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
}


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    text: str = Field(min_length=1, max_length=100_000)
    model: str | None = None
    compare: bool = False


def error(status, message):
    return JSONResponse({'error': message}, status_code=status, headers=HEADERS)


def create_app(service=None, *, deployment=None, request_timeout=REQUEST_TIMEOUT_SECONDS):
    """Inject a service in tests; load the active local models only at startup."""
    @asynccontextmanager
    async def lifespan(app):
        if service is None:
            from kazstyle.inference.service import InferenceService
            app.state.service = InferenceService(deployment=deployment)
        else:
            app.state.service = service
        yield
        app.state.service = None

    # Built-in Swagger/ReDoc pages fetch CDN assets. Keep this application offline.
    app = FastAPI(title='KazStyle', lifespan=lifespan, docs_url=None, redoc_url=None)
    inference_slot = threading.BoundedSemaphore(1)

    @app.middleware('http')
    async def local_request_policy(request: Request, call_next):
        if request.method == 'POST' and request.url.path == '/api/classify':
            port = request.url.port or 80
            origin = request.headers.get('origin')
            if origin and origin not in {f'http://127.0.0.1:{port}', f'http://localhost:{port}'}:
                return error(403, 'Запрос разрешён только с локального сайта.')
            if request.headers.get('content-type', '').split(';')[0].lower() != 'application/json':
                return error(415, 'Ожидается JSON.')
            lengths = request.headers.getlist('content-length')
            if (request.headers.get('transfer-encoding') or len(lengths) != 1
                    or len(lengths[0]) > 8 or not lengths[0].isascii() or not lengths[0].isdigit()):
                return error(400, 'Некорректная длина запроса.')
            length = int(lengths[0])
            if not 0 < length <= MAX_BODY_BYTES:
                return error(413, 'Пустой или слишком большой запрос.')
            try:
                body = await asyncio.wait_for(request.body(), timeout=request_timeout)
            except TimeoutError:
                return error(408, 'Истекло время передачи запроса.')
            except ClientDisconnect:
                return error(400, 'Запрос передан не полностью.')
            if len(body) != length:
                return error(400, 'Запрос передан не полностью.')
        response = await call_next(request)
        response.headers.update(HEADERS)
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_payload(request, exc):
        # Pydantic's detailed errors can contain the pasted text; do not echo it.
        return error(400, 'Некорректный запрос. Проверьте текст, модель и режим сравнения.')

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error(exc.status_code, 'Страница не найдена.' if exc.status_code == 404 else str(exc.detail))

    @app.get('/api/health')
    def health():
        return {'status': 'ready'}

    @app.get('/api/models')
    def models(request: Request):
        current = request.app.state.service
        return {'models': current.available_models(), 'default': current.default_model,
                'styles': current.available_styles(), 'dataset': current.dataset.name}

    @app.post('/api/classify')
    def classify(payload: ClassifyRequest, request: Request):
        # FastAPI runs this synchronous endpoint in its thread pool.
        if not inference_slot.acquire(blocking=False):
            return error(503, 'Модель занята другим запросом. Повторите проверку позже.')
        try:
            return request.app.state.service.classify(payload.text, payload.model, payload.compare)
        except ValueError as exc:
            return error(400, str(exc))
        except Exception:
            logging.exception('Local inference failed')
            return error(500, 'Не удалось выполнить проверку. Подробности в журнале сервера.')
        finally:
            inference_slot.release()

    @app.api_route('/api/{path:path}', methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
                   include_in_schema=False)
    def unknown_api(path: str):
        return error(404, 'Маршрут не найден.')

    app.mount('/', StaticFiles(directory=FRONTEND_DIR, html=True), name='frontend')
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--deployment', help='Optional model registry JSON')
    args = parser.parse_args()
    uvicorn.run(create_app(deployment=args.deployment), host='127.0.0.1', port=args.port,
                access_log=False, http='h11')


if __name__ == '__main__':
    main()
