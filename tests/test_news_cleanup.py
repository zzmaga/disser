import json
from pathlib import Path
import tempfile
import unittest

from kazstyle.data.news_cleanup import build, revise_text, review_flags


class NewsCleanupTests(unittest.TestCase):
    def test_source_timestamp_removed_but_body_time_and_other_sources_survive(self):
        raw = '7 Сәуір, 2026 сағат 23:09 Жаңалық. Кездесу сағат 12:30-да басталды.'
        cleaned, _ = revise_text(raw, 'abai.kz')
        self.assertEqual(cleaned, 'Жаңалық. Кездесу сағат 12:30-да басталды.')
        self.assertEqual(revise_text(raw, 'adilet.zan.kz')[0], raw)
        self.assertEqual(revise_text(cleaned, 'abai.kz')[0], cleaned)

    def test_abai_footer_requires_multiple_signals(self):
        main = 'Мәдениет туралы толық хабар.'
        raw = main+' Ең көп оқылған 1 Басқа хабар 3 күн бұрын 0 пікір Үздік материалдар Жаңалықтар'
        cleaned, log = revise_text(raw, 'abai.kz')
        self.assertEqual(cleaned, main)
        self.assertEqual(log['source_rules'][0]['rule'], 'abai_popular_and_recommended')
        ordinary = 'Ең көп оқылған 1 кітап туралы зерттеу жарияланды.'
        self.assertEqual(revise_text(ordinary, 'abai.kz')[0], ordinary)

    def test_baq_breadcrumb_and_recommendations(self):
        raw = '- Мұқаба - Жаңалықтар - Жаңа кітап. Ең оқылған: - Бірінші хабар - Екінші хабар'
        self.assertEqual(revise_text(raw, 'baq.kz')[0], 'Жаңа кітап.')
        self.assertEqual(revise_text(raw, 'example.org')[0], raw)

    def test_plural_credit_removed_but_substantive_photo_text_preserved(self):
        raw = 'Хабар. Суреттер: Алматы қаласы әкімдігінен алынды. Қалада кітапхана ашылды.'
        self.assertEqual(revise_text(raw, 'abai.kz')[0], 'Хабар. Қалада кітапхана ашылды.')
        meaningful = 'Суреттер зерттеу нәтижелерін көрсетеді. Фото: тарихи дерек.'
        self.assertEqual(revise_text(meaningful, 'abai.kz')[0], meaningful)
        self.assertIn('possible_photo_credit', review_flags(meaningful))

    def test_egemen_caption_removal_preserves_following_sentence(self):
        body = 'Қазақстанда жаңа кітапхана ашылды.'
        for prefix in ['Сурет: ', 'Сурет: almaty.tv ', 'Сурет: ашық дереккөзден ']:
            cleaned, _ = revise_text(prefix+body, 'egemen.kz')
            self.assertEqual(cleaned, body)
            self.assertEqual(revise_text(cleaned, 'egemen.kz')[0], cleaned)
        self.assertEqual(revise_text(body+' (Фото: Stringer / Reuters) Хабар.', 'egemen.kz')[0], body+' Хабар.')
        self.assertEqual(revise_text('Сурет: '+body, 'example.org')[0], 'Сурет: '+body)

    def test_literary_illustration_credit_is_anchored_and_bounded(self):
        body = 'Әңгіме осымен аяқталды.'
        self.assertEqual(revise_text(body+' Көрнекі сурет: А. Дейнека. 1934.', 'adebiportal.kz')[0], body)
        long_body = body+' Көрнекі сурет: '+('Бұл әңгіменің жалғасы. '*20)
        self.assertIn('Көрнекі сурет:', revise_text(long_body, 'adebiportal.kz')[0])

    def test_revision_preserves_inputs_labels_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'input.jsonl'; out = root/'revision'
            row = {'doc_id': 'one', 'style': 'publicistic', 'source_domain': 'abai.kz',
                   'text': '7 Сәуір, 2026 сағат 23:09 Қазақстанда жаңа кітапхана ашылды және оқырмандарға көптеген кітап ұсынылды.',
                   'review_status': 'unreviewed'}
            source.write_text(json.dumps(row, ensure_ascii=False)+'\n', encoding='utf-8')
            before = source.read_bytes(); summary = build(source, out)
            result = json.loads((out/'candidates.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(result['style'], row['style'])
            self.assertEqual(result['review_status'], 'pending_review_after_cleanup')
            self.assertEqual(summary['changed'], 1)
            self.assertFalse(summary['ready_for_training'])
            with self.assertRaises(FileExistsError): build(source, out)

    def test_reserved_or_duplicate_records_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/'input.jsonl'
            row={'doc_id':'one','text':'Қазақша мәтін','style':'scientific','usage_status':'reserved_external'}
            for records in [[row], [{**row,'usage_status':None}]*2]:
                source.write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
                with self.assertRaises(ValueError): build(source,root/'out')
                self.assertFalse((root/'out').exists())


if __name__ == '__main__':
    unittest.main()
