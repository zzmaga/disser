"""Atomic checkpoints for unfinished collection jobs; completed outputs are immutable."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def atomic_json(path, value):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8')
    # The temporary file is on the same filesystem as the destination.
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def state_hash(state):
    return hashlib.sha256(json.dumps(state, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


class CollectionCheckpoint:
    """One atomic state contains both progress cursor and collected records.

    Derived candidates.jsonl may be rebuilt from this state after interruption.
    Protocol (including parser code hashes) must match exactly on resume.
    """
    def __init__(self, out, protocol, initial_state, resume=False):
        self.out = Path(out)
        self.path = self.out/'CHECKPOINT.json'
        if (self.out/'COMPLETE.json').exists():
            raise FileExistsError('Completed collection cannot be resumed')
        if resume:
            envelope = json.loads(self.path.read_text(encoding='utf-8'))
            if envelope.get('schema_version') != 1 or envelope.get('protocol') != protocol:
                raise ValueError('Checkpoint protocol differs; use a new collection directory')
            if state_hash(envelope['state']) != envelope.get('state_sha256'):
                raise ValueError('Checkpoint state checksum mismatch')
            self.state = envelope['state']
        else:
            self.out.mkdir(parents=True, exist_ok=False)
            self.state = initial_state
        self.protocol = protocol
        if not resume:
            self.save('started')

    def save(self, status='running'):
        if (self.out/'COMPLETE.json').exists():
            raise FileExistsError('Completed collection is immutable')
        atomic_json(self.path, {'schema_version':1, 'status':status,
                               'protocol':self.protocol, 'state':self.state,
                               'state_sha256':state_hash(self.state)})


class CollectionPaused(RuntimeError):
    """A deliberate time/request budget stop, not a failed crawl."""
