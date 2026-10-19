"""Run the actual ASGI server on an ephemeral loopback port in transport tests."""
from contextlib import contextmanager
import socket
import threading
import time

import uvicorn


@contextmanager
def running_server(app):
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    address = listener.getsockname()
    config = uvicorn.Config(app, host=address[0], port=address[1], http='h11',
                            access_log=False, log_level='error', timeout_graceful_shutdown=2)
    server = uvicorn.Server(config)
    worker = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and worker.is_alive() and time.monotonic() < deadline:
            time.sleep(.01)
        if not server.started:
            raise RuntimeError('Test ASGI server did not start')
        yield address
    finally:
        server.should_exit = True
        worker.join(timeout=5)
        listener.close()
        if worker.is_alive():
            raise RuntimeError('Test ASGI server did not stop')
