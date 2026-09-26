"""Shared project locations. Importing this module does not load any models."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / 'frontend'


def project_path(relative):
    return PROJECT_ROOT / relative
