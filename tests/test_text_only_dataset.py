"""Data-boundary regressions: forbid metadata features, tampering and parent leakage."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from kazstyle.data.corpus import file_hash, load_manifest
from kazstyle.data.quality import assert_text_only

DATASET = Path('data/processed/text_only_v3')


@unittest.skipUnless((DATASET / 'COMPLETE.json').exists(), 'Requires the built v3 dataset')
class TextOnlyDatasetTests(unittest.TestCase):
    def test_balance_and_parent_separation(self):
        frame, config = load_manifest(DATASET)
        self.assertEqual(len(config['styles']), 5)
        for _, part in frame.groupby('split'):
            self.assertEqual(part.label.nunique(), 5)
            self.assertEqual(part.label.value_counts().nunique(), 1)
        for column in ['parent_group', 'doc_id', 'duplicate_group_id', 'content_hash', 'excerpt_hash']:
            self.assertEqual(int(frame.groupby(column).split.nunique().max()), 1)
        self.assertEqual(set(frame[frame.genre == 'application_form'].split), {'train'})
        assert_text_only(frame.text)
        self.assertTrue(frame.source_url.str.contains('https?://').any())

    def test_changed_csv_and_injected_url_column_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            copied = Path(root)
            for name in ['COMPLETE.json', 'config.json', 'manifest.jsonl', 'metadata.jsonl',
                         'train.csv', 'validation.csv', 'test.csv']:
                shutil.copy2(DATASET / name, copied / name)
            path = copied / 'train.csv'
            inputs = pd.read_csv(path)
            inputs['source_url'] = 'https://adilet.zan.kz'
            inputs.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'input file changed'):
                load_manifest(copied)
            config = json.loads((copied / 'config.json').read_text(encoding='utf-8'))
            config['input_file_hashes']['train.csv'] = file_hash(path)
            (copied / 'config.json').write_text(json.dumps(config), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Only text and label'):
                load_manifest(copied)


if __name__ == '__main__':
    unittest.main()
