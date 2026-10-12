import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from kazstyle.data.corpus import file_hash
from kazstyle.data.customer_reviews import build

TEXT = 'Маған бұл кітап өте ұнады. Кеше кешке оқып бітірдім, енді досыма да оқуға бергім келеді.'


class CustomerReviewImportTests(unittest.TestCase):
    def source(self, root, rows):
        source = root/'source'
        source.mkdir()
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=['custom_id','text','text_cleaned','label','domain'])
        writer.writeheader()
        writer.writerows(rows)
        archive = source/'source.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('source.csv', stream.getvalue().encode('utf8'))
        (source/'DOWNLOAD_COMPLETE.json').write_text(json.dumps({
            'filename':'source.zip','sha256':file_hash(archive),'revision':'pinned',
            'url':'https://example.org/source.zip','license':'CC-BY-4.0','attribution':'Authors',
            'upstream_split':'score_classification_train_imbalanced'}),encoding='utf8')
        return source

    def row(self, ident='a', **changes):
        return {'custom_id':ident,'text':TEXT,'text_cleaned':'UPSTREAM CLEANED TEXT MUST NOT BE USED',
                'label':'4','domain':'bookstore', **changes}

    def test_original_text_not_sentiment_becomes_a_review_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=self.source(root,[self.row(),self.row('duplicate',domain='market')])
            report=build(source,root/'out',1)
            row=json.loads((root/'out/candidates.jsonl').read_text(encoding='utf8'))
            self.assertEqual(row['text'],TEXT)
            self.assertEqual(row['upstream_sentiment_score_zero_based'],4)
            self.assertTrue(row['requires_explicit_style_review'])
            self.assertEqual(row['review_status'],'unreviewed')
            self.assertEqual(report['selected'],1)
            self.assertEqual(report['audit_reasons']['normalized_exact_duplicate'],1)
            with self.assertRaises(FileExistsError):build(source,root/'out',1)

    def test_changed_source_and_resampled_ids_fail_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.source(root,[self.row(),self.row()])
            with self.assertRaisesRegex(ValueError,'Repeated upstream ID'):
                build(source,root/'out',1)
            self.assertFalse((root/'out').exists())
            with (source/'source.zip').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Source hash mismatch'):
                build(source,root/'out',1)

    def test_invalid_sentiment_schema_is_not_silently_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.source(root,[self.row(label='5')])
            with self.assertRaisesRegex(ValueError,'Unexpected source record'):
                build(source,root/'out',1)
            self.assertFalse((root/'out').exists())
