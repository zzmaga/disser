"""Shared project locations. Importing this module does not load any models."""
from pathlib import Path
from kazstyle.experiment_history import evidence_directory, history_index, is_retired

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / 'frontend'


def project_path(relative):
    return PROJECT_ROOT / relative


def artifact_path(run, root=PROJECT_ROOT):
    """Read active runs or compact historical evidence; never choose an output path."""
    if not run or Path(run).name != run or run in {'.', '..'}:
        raise ValueError('Expected a run ID, not a path')
    root = Path(root)
    active = root / 'artifacts' / run
    if is_retired(run, root):
        if active.exists():
            raise ValueError(f'Ambiguous active and retired run: {run}')
        return evidence_directory('artifacts', run, root)
    return active


def evidence_path(relative, root=PROJECT_ROOT):
    """Resolve historical artifact references without rewriting frozen evidence."""
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Expected a project-relative evidence path')
    if len(relative.parts) >= 2 and relative.parts[0] == 'artifacts':
        return artifact_path(relative.parts[1], root).joinpath(*relative.parts[2:])
    if len(relative.parts) >= 2 and relative.parts[0] == 'reports':
        active = Path(root)/'reports'/relative.parts[1]
        if relative.parts[1] in history_index(root).get('reports', []):
            if active.exists():
                raise ValueError(f'Ambiguous report in active and historical storage: {relative.parts[1]}')
            active = evidence_directory('reports', relative.parts[1], root)
        return active.joinpath(*relative.parts[2:])
    return Path(root) / relative


def require_new_run(path, root=PROJECT_ROOT):
    """Do not let a new training run reuse a retired experiment ID."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite run: {path}')
    if path.resolve().parent == (Path(root)/'artifacts').resolve() and is_retired(path.name, root):
        raise FileExistsError(f'Run ID is retired; choose a new experiment ID: {path.name}')
