"""Read compact evidence for retired runs; this bundle never contains model weights."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def history_index(root):
    path = Path(root) / 'configs/retired_experiments.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def is_retired(run, root):
    return run in history_index(root).get('runs', [])


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evidence_directory(kind, name, root):
    """Materialize one checksummed evidence directory in a disposable build cache.

    Original bytes stay in the ZIP. Missing or modified evidence fails explicitly;
    no weight downloads, model reconstruction, or training happens here.
    """
    root = Path(root)
    index = history_index(root)
    names = index.get('runs' if kind == 'artifacts' else 'reports', [])
    if kind not in {'artifacts', 'reports'} or name not in names:
        raise ValueError('Unknown historical evidence directory')
    relative = PurePosixPath(index['bundle'])
    if relative.is_absolute() or '..' in relative.parts or ':' in str(relative) or '\\' in str(relative):
        raise ValueError('Invalid evidence bundle location')
    bundle = root.joinpath(*relative.parts)
    if not bundle.is_file():
        raise FileNotFoundError('Historical evidence ZIP is missing; see docs/EXPERIMENTS.md')
    if _sha(bundle) != index['bundle_sha256']:
        raise ValueError('Historical evidence ZIP checksum changed')
    cache = root / 'build/experiment_evidence' / index['bundle_sha256']
    prefix = f'{kind}/{name}/'
    with zipfile.ZipFile(bundle) as source:
        manifest = json.loads(source.read('MANIFEST.json'))
        selected = {p: sha for p, sha in manifest['files'].items() if p.startswith(prefix)}
        if not selected:
            raise ValueError('Historical evidence directory has no recorded files')
        for member, sha in selected.items():
            parts = PurePosixPath(member)
            if parts.is_absolute() or '..' in parts.parts or ':' in member or '\\' in member:
                raise ValueError('Unsafe historical evidence member')
            target = cache.joinpath(*parts.parts)
            if not target.resolve().is_relative_to(cache.resolve()):
                raise ValueError('Historical evidence cache escapes its directory')
            if target.exists():
                if _sha(target) != sha:
                    raise ValueError('Historical evidence cache was modified')
                continue
            raw = source.read(member)
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError('Historical evidence member checksum changed')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    return cache / kind / name
