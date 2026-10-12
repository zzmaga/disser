import unittest

from kazstyle.data.document_windows import partition_document


class WordTokenizer:
    def __init__(self, factor=1):
        self.factor = factor

    def __call__(self, text, **kwargs):
        assert not kwargs.get('truncation')
        return {'input_ids': [0] * (len(text.split()) * self.factor + 2)}


class DocumentWindowTests(unittest.TestCase):
    def test_all_words_preserved_once_and_offsets_match_original(self):
        text = '  Бірінші сөз\n\nЕкінші  сөз.\tҮшінші сөз, төртінші сөз! Бесінші сөз.  '
        result = partition_document(text, [('first',WordTokenizer(),6), ('second',WordTokenizer(2),8)], 5)
        parts = result['windows']
        self.assertTrue(result['complete'])
        self.assertEqual(result['word_coverage'], 1.0)
        self.assertEqual([word for p in parts for word in p['text'].split()], text.split())
        self.assertEqual([p['words'] for p in parts], [3,3,3,1])
        for part in parts:
            self.assertEqual(part['text'], text[part['char_start']:part['char_end']])
            self.assertLessEqual(part['token_counts']['second'],8)
        for a,b in zip(parts,parts[1:]):
            self.assertEqual(a['word_end'],b['word_start'])
            self.assertLessEqual(a['char_end'],b['char_start'])
        self.assertIsNone(result['next_char_start'])

    def test_bounded_preview_reports_unprocessed_suffix(self):
        text = 'бір екі үш төрт бес алты жеті сегіз тоғыз он'
        result = partition_document(text, [('first',WordTokenizer(),5)], 3, 2)
        self.assertFalse(result['complete'])
        self.assertEqual(result['covered_words'],6)
        self.assertEqual(result['word_coverage'],.6)
        self.assertEqual(text[result['next_char_start']:],'жеті сегіз тоғыз он')

    def test_single_unfittable_word_raises_instead_of_losing_text(self):
        with self.assertRaisesRegex(ValueError,'No complete word'):
            partition_document('мәтін', [('too_small',WordTokenizer(),2)])

    def test_empty_and_short_documents(self):
        empty=partition_document(' \n ', [('first',WordTokenizer(),20)])
        self.assertEqual(empty['windows'],[])
        short=partition_document('бір екі', [('first',WordTokenizer(),20)])
        self.assertEqual(len(short['windows']),1)
        self.assertTrue(short['windows'][0]['short_tail_requires_review'])

    def test_invalid_limits_and_duplicate_tokenizers(self):
        for kwargs in [{'max_words':0},{'max_words':True},{'max_windows':0}]:
            with self.assertRaises(ValueError):
                partition_document('мәтін',[('first',WordTokenizer(),20)],**kwargs)
        with self.assertRaises(ValueError):
            partition_document('мәтін',[('same',WordTokenizer(),20),('same',WordTokenizer(),20)])


if __name__ == '__main__':
    unittest.main()
