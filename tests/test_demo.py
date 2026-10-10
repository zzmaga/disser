import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kazstyle.demo import build, prepare
from kazstyle.settings import PROJECT_ROOT


class DemoTests(unittest.TestCase):
    def test_fixture_preserves_kazakh_and_dates_removes_duplicates_and_html(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'result'
            summary = build(PROJECT_ROOT/'examples/cleaning_input.jsonl', out)
            rows = [json.loads(line) for line in (out/'cleaned.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual((summary['input_rows'], summary['kept_rows'], summary['excluded_rows']), (5, 3, 2))
            self.assertEqual(rows[0]['text'], 'Өсімдіктер жарық энергиясын пайдаланып, органикалық заттар түзеді.')
            self.assertIn('Б. Қалиевке', rows[1]['text'])
            self.assertIn('04.10.2026', rows[1]['text'])
            self.assertEqual(rows[2]['text'], 'Бүгін қалада жаңа кітапхана ашылды.')
            self.assertEqual(summary['excluded'][0]['duplicate_of'], 'science')
            self.assertNotIn('<script>', (out/'report.html').read_text(encoding='utf-8'))

    def test_invalid_schema_and_duplicate_ids_fail(self):
        for records in [[{'id': 'a', 'text': 7}], [{'id': '', 'text': 'мәтін'}],
                        [{'id': 'a', 'text': 'мәтін', 'label': 0}],
                        [{'id': 'a', 'text': 'бір'}, {'id': 'a', 'text': 'екі'}]]:
            with self.subTest(records=records), self.assertRaises(ValueError):
                prepare(records)

    def test_changing_only_url_keeps_output_identical(self):
        text = 'Қазақстанда ғылым мен білім туралы хабар жарияланды.'
        first, _ = prepare([{'id': 'a', 'text': text+' https://example.org/one'}])
        second, _ = prepare([{'id': 'a', 'text': text+' https://example.net/two'}])
        self.assertEqual(first, second)

    def test_refuses_overwrite_and_invalid_input_creates_no_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker = root/'keep.txt';marker.write_text('original', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                build(PROJECT_ROOT/'examples/cleaning_input.jsonl', root)
            bad = root/'bad.jsonl';bad.write_text('{bad}', encoding='utf-8')
            with self.assertRaises(ValueError):
                build(bad, root/'new')
            self.assertFalse((root/'new').exists())
            self.assertEqual(marker.read_text(encoding='utf-8'), 'original')

    def test_cli_outputs_summary_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'cli'
            result = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'kazstyle.demo', '--out-dir', str(out)],
                                    cwd=PROJECT_ROOT, capture_output=True, text=True, encoding='utf-8', timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['kept_rows'], 3)
            self.assertTrue((out/'report.html').is_file())


if __name__ == '__main__':
    unittest.main()
