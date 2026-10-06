"""Shared project locations. Importing this module does not load any models."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / 'frontend'


def project_path(relative):
    return PROJECT_ROOT / relative


def artifact_path(run, root=PROJECT_ROOT):
    """Read a run from active storage or the cold archive; never choose an output path."""
    if not run or Path(run).name != run or run in {'.', '..'}:
        raise ValueError('Expected a run ID, not a path')
    root = Path(root)
    active = root / 'artifacts' / run
    archived = root / 'archive' / 'artifacts' / run
    if active.exists() and archived.exists():
        raise ValueError(f'Ambiguous run in active storage and archive: {run}')
    return archived if archived.exists() else active


def evidence_path(relative, root=PROJECT_ROOT):
    """Resolve historical artifact references without rewriting frozen evidence."""
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Expected a project-relative evidence path')
    if len(relative.parts) >= 2 and relative.parts[0] == 'artifacts':
        return artifact_path(relative.parts[1], root).joinpath(*relative.parts[2:])
    if len(relative.parts) >= 2 and relative.parts[0] == 'reports':
        active = Path(root)/'reports'/relative.parts[1]
        archived = Path(root)/'archive/reports'/relative.parts[1]
        if active.exists() and archived.exists():
            raise ValueError(f'Ambiguous report in active storage and archive: {relative.parts[1]}')
        return (archived if archived.exists() else active).joinpath(*relative.parts[2:])
    return Path(root) / relative


def require_new_run(path, root=PROJECT_ROOT):
    """Do not let a new training run reuse an archived experiment ID."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite run: {path}')
    if path.resolve().parent == (Path(root)/'artifacts').resolve() and artifact_path(path.name, root).exists():
        raise FileExistsError(f'Run ID already exists in the archive: {path.name}')
