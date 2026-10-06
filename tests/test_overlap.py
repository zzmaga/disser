import json
import tempfile
import unittest
from pathlib import Path

from kazstyle.data.corpus import write_jsonl
from kazstyle.data.overlap import audit, words, overlap_metrics


class PassageOverlapTests(unittest.TestCase):
    def test_short_questions_with_different_headings_still_flagged(self):
        result = overlap_metrics(10, 23, 15)
        self.assertEqual(result['flag'], 'shared_passage')
        self.assertIsNone(overlap_metrics(4, 23, 15))

    def test_contained_excerpt_and_punctuation_variants_are_detected(self):
        tokens = [f'сөз{i}' for i in range(100)]
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            candidates, references = root / 'new.jsonl', root / 'old.jsonl'
            write_jsonl(candidates, [
                {'doc_id': 'excerpt', 'text': ' '.join(tokens[20:40]), 'style': 'literary'},
                {'doc_id': 'variant', 'text': ', '.join(tokens[20:40]).upper(), 'style': 'literary'},
                {'doc_id': 'unrelated', 'text': 'өзге бөлек мәтін', 'style': 'colloquial'}])
            write_jsonl(references, [{'doc_id': 'long', 'text': ' '.join(tokens), 'style': 'literary'}])
            summary = audit([candidates], [references], root / 'out')
            self.assertEqual(summary['by_flag'], {'normalized_exact': 1, 'high_containment': 2})
            self.assertEqual(summary['flagged_candidates'], 2)
            findings = [json.loads(x) for x in (root/'out/findings.jsonl').read_text().splitlines()]
            pair = next(r for r in findings if r['right']['doc_id'] == 'long')
            self.assertLess(pair['jaccard'], .2)
            self.assertEqual(pair['containment'], 1.)
            with self.assertRaises(FileExistsError):
                audit([candidates], [references], root / 'out')

    def test_digits_preserved_and_short_exact_texts_checked(self):
        self.assertNotEqual(words('№ 42'), words('№ 43'))
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); source = root / 'input.jsonl'
            write_jsonl(source, [{'doc_id': 'a', 'text': 'Сәлем, әлем!'},
                                 {'doc_id': 'b', 'text': 'сәлем әлем'}])
            result = audit([source], [], root / 'out')
            self.assertEqual(result['by_flag'], {'normalized_exact': 1})


if __name__ == '__main__':
    unittest.main()
