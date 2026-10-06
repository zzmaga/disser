import unittest
import pandas as pd
from kazstyle.data.corpus import digest
from kazstyle.data.morphology import aligned_annotations, project_words


class MorphologyTests(unittest.TestCase):
    def test_distinguishes_surface_from_lemmas_without_silent_fallback(self):
        words = [{'text': 'Балалар', 'lemma': 'бала'}, {'text': 'келді', 'lemma': 'кел'}, {'text': '.', 'lemma': '.'}]
        self.assertEqual(project_words(words, 'text'), 'Балалар келді .')
        self.assertEqual(project_words(words, 'lemma'), 'бала кел .')
        with self.assertRaises(ValueError):
            project_words([{'text': 'Белгісіз', 'lemma': None}], 'lemma')

    def test_rejects_changed_text_duplicates_and_wrong_parent(self):
        frame = pd.DataFrame([{'sample_id': 's1', 'doc_id': 'd1', 'text': 'Мәтін.'}])
        row = {'sample_id': 's1', 'doc_id': 'd1', 'text_sha256': digest('Мәтін.'),
               'words': [{'text': 'Мәтін', 'lemma': 'мәтін'}]}
        self.assertEqual(aligned_annotations(frame, [row]), [row])
        for bad in [[row, row], [], [{**row, 'doc_id': 'wrong'}], [{**row, 'text_sha256': digest('Басқа мәтін')}]]:
            with self.assertRaises(ValueError):
                aligned_annotations(frame, bad)
