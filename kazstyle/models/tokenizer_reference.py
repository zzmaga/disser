"""Freeze tokenizer constraints independently of a training/evaluation dataset."""
import argparse
import json
import re
import shutil
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from kazstyle.data.corpus import file_hash, write_json
from kazstyle.models.tokenization import load_dataset_tokenizer


def build(protocol_path, out, *, max_tokens=256, allow_download=False):
    protocol_path, out = Path(protocol_path), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if max_tokens < 16:
        raise ValueError('A meaningful encoder input limit is required')
    source_sha, protocol_sha = file_hash(Path(__file__)), file_hash(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    specs = [r for r in protocol['primary_models'] if r.get('model_name')]
    if not specs or len({r['model_name'] for r in specs}) != len(specs):
        raise ValueError('Register each encoder tokenizer exactly once')
    if any(not re.fullmatch('[0-9a-f]{40}', r.get('revision', '')) for r in specs):
        raise ValueError('Encoder tokenizers require pinned commit revisions')
    loaded = []
    for spec in specs:
        args = {'revision': spec['revision'], 'local_files_only': not allow_download,
                'trust_remote_code': False}
        config = AutoConfig.from_pretrained(spec['model_name'], **args)
        if config._commit_hash != spec['revision']:
            raise ValueError('Resolved encoder revision differs from protocol')
        tokenizer = AutoTokenizer.from_pretrained(spec['model_name'], **args)
        loaded.append((spec, config, tokenizer))
    if file_hash(protocol_path) != protocol_sha or file_hash(Path(__file__)) != source_sha:
        raise ValueError('Tokenizer reference inputs changed during preparation')
    out.mkdir(parents=True)
    snapshots = []
    for i, (spec, encoder, tokenizer) in enumerate(loaded):
        relative = 'tokenizer' if i == 0 else f'tokenizers/encoder_{i}'
        tokenizer.save_pretrained(out/relative)
        encoder.save_pretrained(out/f'encoder_configs/{i}')
        restored = AutoTokenizer.from_pretrained(out/relative, local_files_only=True)
        probe = 'Қазақ тіліндегі мәтіннің стилін анықтау.'
        if tokenizer(probe)['input_ids'] != restored(probe)['input_ids']:
            raise ValueError('Tokenizer snapshot changed encoding')
        snapshots.append({'model_name': spec['model_name'], 'revision': spec['revision'],
                          'path': relative, 'max_tokens': max_tokens,
                          'hashes': {p.name: file_hash(p) for p in (out/relative).iterdir() if p.is_file()}})
    first = snapshots[0]
    config = {'purpose': 'tokenizer_constraint_reference_NOT_training_dataset',
              'protocol_sha256': protocol_sha, 'max_tokens': max_tokens,
              'tokenizer_name': first['model_name'], 'tokenizer_revision': first['revision'],
              'tokenizer_hashes': first['hashes'], 'additional_tokenizers': snapshots[1:],
              'encoder_weights_downloaded_by_this_command': False}
    write_json(out/'config.json', config)
    shutil.copy2(protocol_path, out/'protocol.json'); shutil.copy2(__file__, out/'code_snapshot.py')
    if file_hash(out/'protocol.json') != protocol_sha or file_hash(out/'code_snapshot.py') != source_sha:
        raise ValueError('Reference source changed while taking snapshot')
    for spec in snapshots:
        load_dataset_tokenizer(out, config, spec['model_name'])
    write_json(out/'COMPLETE.json', {p.relative_to(out).as_posix(): file_hash(p)
                                  for p in sorted(out.rglob('*')) if p.is_file()})
    return {'reference': str(out), 'tokenizers': len(snapshots), 'max_tokens_each': max_tokens,
            'model_names': [r['model_name'] for r in specs], 'training_performed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--allow-download', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(args.protocol, args.out_dir, max_tokens=args.max_tokens,
                           allow_download=args.allow_download)))
