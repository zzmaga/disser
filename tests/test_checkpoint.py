import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kazstyle.data.checkpoint import CollectionCheckpoint, atomic_json


class CheckpointTests(unittest.TestCase):
    def test_resume_restores_cursor_and_rows_and_rejects_changed_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'job'; protocol = {'limit':10,'code_sha':'fixed'}
            job = CollectionCheckpoint(out, protocol, {'cursor':0,'rows':[]})
            job.state.update(cursor=3,rows=[{'text':'қазақша'}]);job.save('paused')
            restored = CollectionCheckpoint(out, protocol, {}, resume=True)
            self.assertEqual(restored.state,job.state)
            with self.assertRaisesRegex(ValueError,'protocol'):
                CollectionCheckpoint(out, {**protocol,'limit':11}, {}, resume=True)
            (out/'COMPLETE.json').write_text('{}')
            with self.assertRaises(FileExistsError):job.save()
            with self.assertRaises(FileExistsError):CollectionCheckpoint(out,protocol,{},resume=True)

    def test_interrupted_replace_preserves_last_checkpoint_and_removes_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.json';atomic_json(path,{'cursor':1})
            before=path.read_bytes()
            with patch('kazstyle.data.checkpoint.os.replace',side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):atomic_json(path,{'cursor':2})
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(list(Path(tmp).iterdir()),[path])

    def test_corrupt_state_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'job';job=CollectionCheckpoint(out,{}, {'rows':[]})
            envelope=json.loads(job.path.read_text(encoding='utf-8'))
            envelope['state']['rows'].append('changed')
            job.path.write_text(json.dumps(envelope),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'checksum'):
                CollectionCheckpoint(out,{}, {}, resume=True)


if __name__=='__main__':unittest.main()
