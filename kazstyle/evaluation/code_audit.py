"""Verify the computational source snapshots of a completed experiment suite."""
import argparse
import json
from pathlib import Path

from kazstyle.data.corpus import file_hash, load_manifest, write_json
from kazstyle.settings import PROJECT_ROOT, artifact_path

COMPUTATIONAL_FILES = (
    'kazstyle/training/classical.py', 'kazstyle/training/transformer.py',
    'kazstyle/models/style_transformer.py', 'kazstyle/models/tokenization.py',
    'kazstyle/data/corpus.py', 'kazstyle/data/quality.py',
    'kazstyle/evaluation/reports.py',
)


def audit(suite, plan_path, out):
    if out.exists():
        raise FileExistsError(out)
    protocol = json.loads((suite/'protocol.json').read_text(encoding='utf-8'))
    status = json.loads((suite/'status.json').read_text(encoding='utf-8'))
    if status['status'] != 'complete' or len(status['completed']) != status['total']:
        raise ValueError('Every planned run must be complete')
    if file_hash(plan_path) != protocol['plan_sha256']:
        raise ValueError('The experiment plan changed')
    _, config = load_manifest(PROJECT_ROOT/protocol['plan']['dataset'])
    versions = {path: set() for path in COMPUTATIONAL_FILES}
    runs = {}
    for run_id in status['completed']:
        run = artifact_path(run_id)
        provenance = json.loads((run/'provenance.json').read_text(encoding='utf-8'))
        marker = json.loads((run/'COMPLETE.json').read_text(encoding='utf-8'))
        if marker['plan_sha256'] != protocol['plan_sha256']:
            raise ValueError('Wrong run plan')
        if provenance['manifest_sha256'] != config['manifest_sha256']:
            raise ValueError('Wrong dataset')
        # Verify every archived Python file, including supporting modules.
        for relative, expected in provenance['source_sha256'].items():
            if file_hash(run/'source_snapshot'/relative) != expected:
                raise ValueError(f'Changed snapshot: {run_id}/{relative}')
        for relative in COMPUTATIONAL_FILES:
            versions[relative].add(provenance['source_sha256'][relative])
        runs[run_id] = {'verified_snapshot_files': len(provenance['source_sha256']),
                        'provenance_sha256': file_hash(run/'provenance.json')}
    if any(len(hashes) != 1 for hashes in versions.values()):
        raise ValueError('Computational code differed across the declared runs')
    record = {'plan_sha256': protocol['plan_sha256'],
              'manifest_sha256': config['manifest_sha256'], 'runs': runs,
              'identical_computational_files': {p: next(iter(h)) for p, h in versions.items()},
              'scope': 'Archived snapshots verified; later working-tree improvements do not rewrite the experiment.'}
    write_json(out, record)
    print(json.dumps({'runs': len(runs), 'identical_computational_files': len(versions), 'report': str(out)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out-file', type=Path, required=True)
    a = p.parse_args()
    audit(a.suite, a.plan, a.out_file)
