@echo off
cd /d "%~dp0"
echo.
echo Open in your browser: http://127.0.0.1:8765
echo Keep this window open while testing. Press Ctrl+C to stop.
echo.
".venv\Scripts\python.exe" manage.py serve --port 8765
if errorlevel 1 pause
