import tempfile
import unittest
from pathlib import Path
from kazstyle.settings import artifact_path, evidence_path, require_new_run


class StorageTests(unittest.TestCase):
    def test_historical_report_paths_resolve_after_archiving(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);archived=root/'archive/reports/study'
            archived.mkdir(parents=True)
            self.assertEqual(evidence_path('reports/study/protocol.json',root),archived/'protocol.json')
            (root/'reports/study').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'Ambiguous report'):
                evidence_path('reports/study/protocol.json',root)

    def test_archived_run_retains_original_evidence_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archived = root/'archive/artifacts/experiment_s43'
            archived.mkdir(parents=True)
            (archived/'results.json').write_text('{}')
            self.assertEqual(artifact_path('experiment_s43', root), archived)
            self.assertEqual(evidence_path('artifacts/experiment_s43/results.json', root), archived/'results.json')
            with self.assertRaises(FileExistsError):
                require_new_run(root/'artifacts/experiment_s43', root)
            require_new_run(root/'artifacts/new_unique_id', root)
            active = root/'artifacts/experiment_s43'
            active.mkdir(parents=True)
            with self.assertRaises(ValueError):
                artifact_path('experiment_s43', root)

    def test_missing_run_stays_in_active_storage_and_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(artifact_path('new_run', root), root/'artifacts/new_run')
            for value in ['../old', 'artifacts/old', '..', '']:
                with self.assertRaises(ValueError):
                    artifact_path(value, root)
            with self.assertRaises(ValueError):
                evidence_path('../outside/results.json', root)
