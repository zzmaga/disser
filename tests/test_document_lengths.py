import csv
from pathlib import Path
import tempfile
import unittest

from kazstyle.evaluation.document_lengths import csv_structure, length_record


class DocumentLengthTests(unittest.TestCase):
    def test_multiline_quoted_document_is_one_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'corpus.csv'
            with path.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream, lineterminator='\n')
                writer.writerow(['text', 'label'])
                writer.writerow(['Қазақша "мәтін", бірінші жол\nЕкінші жол', 'Artistic'])
                writer.writerow(['Қысқа мәтін', 'Colloquial'])
            result = csv_structure(path)
            self.assertTrue(result['valid_structure'])
            self.assertEqual((result['records'], result['physical_lines'], result['multiline_records']), (2,4,1))

    def test_unclosed_quote_is_not_counted_as_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.csv'
            path.write_text('text,label\n"unfinished\nFormal\n', encoding='utf-8')
            self.assertFalse(csv_structure(path)['valid_structure'])

    def test_extra_column_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.csv'
            path.write_text('text,label\ntext,Formal,extra\n', encoding='utf-8')
            result = csv_structure(path)
            self.assertEqual(result['invalid_width'], [{'record':1,'fields':3}])

    def test_actual_newline_differs_from_literal_escape(self):
        result = length_record({'doc_id':'a','style':'literary','text':'Бірінші жол\nЕкінші жол \\n'})
        self.assertEqual(result['nonempty_lines'], 2)
        self.assertEqual(result['literal_backslash_n'], 1)


if __name__ == '__main__':
    unittest.main()
