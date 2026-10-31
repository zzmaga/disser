FROM python:3.12.12-slim@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY configs/requirements-web.txt configs/requirements-container.txt /app/configs/
RUN python -m pip install --no-cache-dir -r configs/requirements-container.txt \
    && python -m pip check
COPY kazstyle/__init__.py kazstyle/settings.py kazstyle/experiment_history.py kazstyle/demo.py /app/kazstyle/
COPY kazstyle/api/__init__.py kazstyle/api/preparation.py /app/kazstyle/api/
COPY kazstyle/data/__init__.py kazstyle/data/quality.py /app/kazstyle/data/
COPY examples/cleaning_input.jsonl /app/examples/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["python", "-m", "uvicorn", "kazstyle.api.preparation:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM runtime AS test
USER root
RUN python -m pip install --no-cache-dir pytest==9.1.1 httpx2==2.14.0
COPY tests/test_demo.py tests/test_preparation_api.py /app/tests/
USER 10001:10001
RUN python -m pytest -q -p no:cacheprovider tests/test_demo.py tests/test_preparation_api.py

FROM runtime AS production
