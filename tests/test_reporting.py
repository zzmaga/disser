"""Reported metrics survive templating; missing evidence and HTML injection fail safely."""
from pathlib import Path
import tempfile
import unittest

from bs4 import BeautifulSoup
from jinja2 import UndefinedError
import pandas as pd

from kazstyle.evaluation.rendering import render_template, table
from kazstyle.evaluation.reports import metrics, provenance, render_report


class ReportingTests(unittest.TestCase):
    def test_table_escapes_text_and_preserves_kazakh(self):
        result = str(table(['Стиль'], [['Ғылыми <script>alert(1)</script>']]))
        self.assertNotIn('<script>', result)
        self.assertIn('&lt;script&gt;', result)
        self.assertIn('Ғылыми', result)

    def test_missing_manuscript_values_are_an_error(self):
        with self.assertRaises(UndefinedError):
            render_template('thesis_english.html')

    def test_report_preserves_metrics_counts_and_confusion_matrix(self):
        config = {'styles':['scientific','official'], 'id_to_label':{'0':'Scientific','1':'Formal'},
                  'manifest_sha256':'fixture-sha', 'limitations':['Preliminary <labels>']}
        result = metrics([0,0,1,1], [0,1,1,1], config)
        self.assertEqual(result['confusion_matrix'], [[1,1],[0,2]])
        frame = pd.DataFrame([{'split':split, 'style_name':name} for split in ['train','validation','test']
                              for name in ['Scientific','Formal']])
        values = {'model <unsafe>':{'validation':result, 'test':result, 'fit_seconds':1.234}}
        with tempfile.TemporaryDirectory() as tmp:
            render_report(Path(tmp), frame, config, values, title='Report <title>')
            html = (Path(tmp)/'report.html').read_text(encoding='utf-8')
            page = BeautifulSoup(html, 'html.parser')
            self.assertEqual(page.title.text, 'Report <title>')
            self.assertNotIn('<unsafe>', html)
            self.assertIn('Preliminary &lt;labels&gt;', html)
            self.assertIn(f"{result['macro_f1']:.4f}", page.get_text())
            matrix = page.select('table')[-1]
            cells = [int(cell.text) for cell in matrix.select('td')]
            self.assertEqual(cells, [1,1,0,2])
            self.assertIn('fixture-sha', page.get_text())
            self.assertIn('0.7333', (Path(tmp)/'report.md').read_text(encoding='utf-8'))

    def test_new_experiment_provenance_includes_templates_and_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            (dataset/'manifest.jsonl').write_text('{}\n', encoding='utf-8')
            record = provenance(dataset, {})
        self.assertIn('kazstyle/evaluation/templates/experiment.html', record['source_sha256'])
        self.assertIn('configs/requirements-web.txt', record['source_sha256'])
        self.assertIn('Jinja2', record['packages'])
