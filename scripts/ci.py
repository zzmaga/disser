"""Run only portable CI tests; failures, empty discovery and skips are errors."""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', action='store_true', help='Also run offline data pipeline tests (requirements-data-ci.txt)')
    args = parser.parse_args()
    os.chdir(ROOT)
    suite = unittest.TestSuite()
    patterns = ['test_quality.py', 'test_storage.py', 'test_demo.py', 'test_news_cleanup.py',
                'test_api_contract.py', 'test_document_lengths.py', 'test_checkpoint.py']
    if args.data:
        patterns += ['test_export_corpus.py', 'test_style_expansion.py', 'test_news_html.py',
                     'test_refresh_news.py', 'test_forum.py', 'test_forum_replies.py', 'test_prose.py', 'test_business.py',
                     'test_pool.py', 'test_grouping.py', 'test_passage_grouping.py', 'test_overlap.py',
                     'test_document_windows.py', 'test_forum_checkpoint.py', 'test_window_probe.py',
                     'test_customer_reviews.py','test_folktales.py','test_curation.py','test_reservation.py']
    for pattern in patterns:
        discovered = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern=pattern)
        if discovered.countTestCases() == 0:
            raise RuntimeError(f'No tests discovered for required pattern: {pattern}')
        suite.addTests(discovered)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    log = stream.getvalue()
    out = ROOT/'build';out.mkdir(exist_ok=True)
    (out/'ci-tests.txt').write_text(log, encoding='utf-8')
    success = result.wasSuccessful() and not result.skipped and result.testsRun >= 28
    summary = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'skipped': len(result.skipped), 'success': success, 'patterns': patterns,
               'scope': 'Offline data/API checks' if args.data else 'Portable cleaning, storage, API and demo checks',
               'trained_model_evaluation': False}
    (out/'ci-tests.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(log)
    print(json.dumps(summary))
    step_summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if step_summary:
        with open(step_summary, 'a', encoding='utf-8') as f:
            f.write(f'## KazStyle portable checks\n\nTests: **{result.testsRun}**; failures: **{len(result.failures)}**; errors: **{len(result.errors)}**; skipped: **{len(result.skipped)}**.\n\nNo GPU, model weights or private datasets are used. This run checks software behavior, not classifier accuracy.\n')
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
