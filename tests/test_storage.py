import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from kazstyle.settings import artifact_path, evidence_path, require_new_run


def fixture(root):
    files = {'artifacts/experiment_s43/results.json': b'{"metric":0.75}',
             'reports/study/protocol.json': b'{"seed":43}'}
    bundle = root/'reports/experiment_history/evidence.zip'
    bundle.parent.mkdir(parents=True)
    with zipfile.ZipFile(bundle, 'w') as z:
        for name, content in files.items():
            z.writestr(name, content)
        z.writestr('MANIFEST.json', json.dumps({'files': {
            name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}))
    (root/'configs').mkdir()
    (root/'configs/retired_experiments.json').write_text(json.dumps({
        'bundle': 'reports/experiment_history/evidence.zip',
        'bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
        'runs': ['experiment_s43'], 'reports': ['study']}), encoding='utf-8')
    return bundle


class StorageTests(unittest.TestCase):
    def test_historical_report_paths_resolve_from_compact_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixture(root)
            path=evidence_path('reports/study/protocol.json',root)
            self.assertEqual(path.read_bytes(),b'{"seed":43}')
            self.assertTrue(path.is_relative_to(root/'build'))
            (root/'reports/study').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'Ambiguous report'):
                evidence_path('reports/study/protocol.json',root)

    def test_retired_run_keeps_results_but_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixture(root)
            retired=artifact_path('experiment_s43',root)
            self.assertEqual((retired/'results.json').read_bytes(),b'{"metric":0.75}')
            self.assertFalse((retired/'best_model.pt').exists())
            self.assertEqual(evidence_path('artifacts/experiment_s43/results.json',root),retired/'results.json')
            with self.assertRaises(FileExistsError):require_new_run(root/'artifacts/experiment_s43',root)
            require_new_run(root/'artifacts/new_unique_id',root)
            (root/'artifacts/experiment_s43').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'Ambiguous'):artifact_path('experiment_s43',root)

    def test_missing_run_stays_in_active_storage_and_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.assertEqual(artifact_path('new_run',root),root/'artifacts/new_run')
            for value in ['../old','artifacts/old','..','']:
                with self.assertRaises(ValueError):artifact_path(value,root)
            with self.assertRaises(ValueError):evidence_path('../outside/results.json',root)

    def test_bundle_corruption_is_rejected_even_when_cache_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);bundle=fixture(root)
            artifact_path('experiment_s43',root)
            with bundle.open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'ZIP checksum'):artifact_path('experiment_s43',root)

    def test_changed_cache_is_not_silently_trusted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);fixture(root)
            run=artifact_path('experiment_s43',root)
            (run/'results.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'cache was modified'):artifact_path('experiment_s43',root)

    def test_missing_bundle_does_not_release_retired_run_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);bundle=fixture(root);bundle.unlink()
            with self.assertRaises(FileNotFoundError):artifact_path('experiment_s43',root)
            with self.assertRaises(FileExistsError):require_new_run(root/'artifacts/experiment_s43',root)

    def test_unsafe_bundle_member_cannot_write_outside_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);bundle=fixture(root)
            member='artifacts/experiment_s43/../../escape.txt'
            with zipfile.ZipFile(bundle,'w') as z:
                z.writestr(member,b'bad')
                z.writestr('MANIFEST.json',json.dumps({'files':{member:hashlib.sha256(b'bad').hexdigest()}}))
            path=root/'configs/retired_experiments.json';index=json.loads(path.read_text())
            index['bundle_sha256']=hashlib.sha256(bundle.read_bytes()).hexdigest()
            path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError,'Unsafe'):artifact_path('experiment_s43',root)
            self.assertFalse((root/'build/escape.txt').exists())
