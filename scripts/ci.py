"""Run only portable CI tests; failures, empty discovery and skips are errors."""
import io
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.chdir(ROOT)
    suite = unittest.TestSuite()
    patterns = ['test_quality.py', 'test_storage.py', 'test_demo.py']
    for pattern in patterns:
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern=pattern))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    log = stream.getvalue()
    out = ROOT/'build';out.mkdir(exist_ok=True)
    (out/'ci-tests.txt').write_text(log, encoding='utf-8')
    success = result.wasSuccessful() and not result.skipped and result.testsRun >= 14
    summary = {'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'skipped': len(result.skipped), 'success': success, 'patterns': patterns,
               'scope': 'Portable cleaning, storage and demo tests; no trained-model evaluation'}
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
