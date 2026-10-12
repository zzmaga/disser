import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from kazstyle.data.corpus import digest, file_hash
from kazstyle.data.export_corpus import build, verify_export, publisher_groups, load_documents, DOMAIN_LIKE, read_recipe

TEXT = 'Қазақстанда жаңа кітапхана ашылды, онда «Қазақ әдебиеті» кітаптары оқырмандарға ұсынылады және балаларға арналған үйірмелер жұмыс істейді.'


class ExportCorpusTests(unittest.TestCase):
    def test_collection_target_is_independent_of_historical_article(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, recipe = self.fixture(root, [self.row('a')])
            result = build(recipe, root/'out')
            self.assertEqual(result['collection_targets']['publicistic'], 2000)
            self.assertEqual(result['shortfall_by_style']['publicistic'], 1999)
            config = json.loads(recipe.read_text())
            for invalid in [{}, {'publicistic': 2000}, {s: True for s in result['collection_targets']}]:
                config['collection_targets'] = invalid
                recipe.write_text(json.dumps(config), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Collection targets'):
                    read_recipe(recipe)

    def test_review_removals_require_exact_boundaries_and_preserve_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = '(әңгіме) '+TEXT+' Сурет ғаламтордан алынды'
            source, recipe = self.fixture(root, [self.row('a', text=original)])
            original_bytes = source.read_bytes()
            review = root/'review.json'
            decision = {'doc_id':'a', 'text_sha256':digest(original), 'keep':True,
                        'reason':'Explicit boundary credits', 'reviewed_extent':'title_beginning_end',
                        'remove_prefix':'(әңгіме) ', 'remove_suffix':' Сурет ғаламтордан алынды'}
            config = json.loads(recipe.read_text())

            def set_decision(value):
                review.write_text(json.dumps({'reviewer_kind':'assistant', 'decisions':[value]}), encoding='utf-8')
                config['reviews'] = [{'path':str(review), 'sha256':file_hash(review)}]
                recipe.write_text(json.dumps(config), encoding='utf-8')

            set_decision(decision)
            build(recipe, root/'out')
            self.assertEqual(load_documents(root/'out')[0]['text'], TEXT)
            self.assertEqual(source.read_bytes(), original_bytes)
            changes = json.loads((root/'out/changes.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(changes['before_sha256'], digest(original))
            self.assertEqual(len([r for r in changes['source_rules'] if r['rule'].startswith('hash_bound')]), 2)
            for invalid in [{**decision, 'remove_prefix':'different'},
                            {**decision, 'remove_suffix':'different'},
                            {**decision, 'keep':False},
                            {**decision, 'text_sha256':'stale'}]:
                set_decision(invalid)
                with self.assertRaises(ValueError):
                    build(recipe, root/'invalid')
                self.assertFalse((root/'invalid').exists())

    def test_publisher_byline_is_a_group_hint_not_a_model_input(self):
        row = {'source_domain':'adebiportal.kz', 'title':'Роза Мұқанова. Үркек керім'}
        first = publisher_groups(row)
        second = publisher_groups({**row, 'title':'Роза Мұқанова. Үркек керім (әңгіме)'})
        self.assertEqual(first['work_group'], second['work_group'])
        self.assertEqual(publisher_groups({**row, 'title':'Әдебиетіміздегі әкелер образы'}), {})
        self.assertEqual(publisher_groups({**row, 'source_domain':'example.org'}), {})
        self.assertNotIn('text', first)

    def test_unrecognized_domains_flagged_without_treating_sentences_as_urls(self):
        self.assertIsNotNone(DOMAIN_LIKE.search('polese.by'))
        self.assertIsNotNone(DOMAIN_LIKE.search('Alash_11 .ru'))
        self.assertIsNone(DOMAIN_LIKE.search('STEM. This is a sentence. 0.25'))

    def fixture(self, root, rows, **extra):
        source = root/'source.jsonl'
        source.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
        recipe = {'purpose': 'provisional_document_corpus',
                  'inputs': [{'path': str(source), 'sha256': file_hash(source)}], **extra}
        path = root/'recipe.json'; path.write_text(json.dumps(recipe), encoding='utf-8')
        return source, path

    def row(self, doc_id, **extra):
        return {'doc_id': doc_id, 'text': TEXT, 'style': 'publicistic',
                'source_domain': 'example.org', 'source_url': 'https://example.org/'+doc_id,
                'parent_id': doc_id, 'review_status': 'unreviewed', **extra}

    def test_csv_roundtrip_provenance_and_no_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, recipe = self.fixture(root, [self.row('a', text=TEXT+'\nhttps://example.org')])
            before = source.read_bytes(); out = root/'out'; result = build(recipe, out)
            self.assertEqual(result['accepted_provisional'], 1)
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse(result['ready_for_training'])
            with (out/'corpus.csv').open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream); self.assertEqual(reader.fieldnames, ['text', 'label'])
                row = next(reader)
            self.assertEqual(row['text'], TEXT)
            meta = json.loads((out/'metadata.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(meta['source_url'], 'https://example.org/a')
            self.assertEqual(meta['text_sha256'], digest(row['text']))
            self.assertEqual(meta['review_status'], 'pending_review_after_cleanup')
            loaded = load_documents(out)
            self.assertEqual(loaded[0]['text'], TEXT)
            self.assertEqual(loaded[0]['source_url'], 'https://example.org/a')
            with self.assertRaises(FileExistsError): build(recipe, out)
            meta['text_sha256'] = 'broken'
            (out/'metadata.jsonl').write_text(json.dumps(meta)+'\n', encoding='utf-8')
            with self.assertRaises(ValueError): verify_export(out)
            with self.assertRaises(ValueError): load_documents(out)

    def test_all_conflicting_labels_excluded_regardless_input_order(self):
        for styles in [('literary', 'scientific'), ('scientific', 'literary')]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); _, recipe = self.fixture(root, [self.row(str(i), style=s) for i, s in enumerate(styles)])
                result = build(recipe, root/'out')
                self.assertEqual(result['accepted_provisional'], 0)
                self.assertEqual(result['excluded_reasons']['conflicting_style_for_same_text'], 2)

    def test_duplicate_keeps_origins_and_uncertain_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [self.row('a'), self.row('b', text=TEXT.upper()),
                    self.row('c', text=TEXT+' Фото: тарихи дерек.')]
            _, recipe = self.fixture(root, rows)
            result = build(recipe, root/'out')
            self.assertEqual((result['accepted_provisional'], result['excluded'], result['review']), (1, 1, 1))
            meta = json.loads((root/'out/metadata.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(meta['duplicate_occurrences'][0]['doc_id'], 'b')

    def test_invalid_hash_reserved_id_and_schema_fail_before_output(self):
        bad_rows = [[self.row('a', usage_status='reserved_external_test')],
                    [self.row('a', style='unknown')], [self.row('')],
                    [self.row('a'), self.row('a', text=TEXT+' Қосымша мәтін.')]]
        for rows in bad_rows:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); _, recipe = self.fixture(root, rows)
                with self.assertRaises(ValueError): build(recipe, root/'out')
                self.assertFalse((root/'out').exists())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, recipe = self.fixture(root, [self.row('a')])
            source.write_text('', encoding='utf-8')
            with self.assertRaises(ValueError): build(recipe, root/'out')
            self.assertFalse((root/'out').exists())

    def test_explicit_exclusion_and_new_genre_review_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, recipe = self.fixture(root, [self.row('a')], exclusions=[
                {'doc_id':'a', 'text_sha256':digest(TEXT), 'reason':'Known wrong genre'}])
            self.assertEqual(build(recipe, root/'excluded')['excluded'], 1)
            config = json.loads(recipe.read_text()); config['exclusions'] = []
            config['inputs'][0]['require_review'] = True
            recipe.write_text(json.dumps(config), encoding='utf-8')
            result = build(recipe, root/'review')
            self.assertEqual((result['accepted_provisional'], result['review']), (0, 1))
            config['inputs'][0]['require_review'] = False
            config['require_review_filters'] = [{'source_domain':'example.org', 'style':'publicistic'}]
            recipe.write_text(json.dumps(config), encoding='utf-8')
            self.assertEqual(build(recipe, root/'source_review')['review'], 1)
            config['require_review_filters'] = [{'source_domain':'different.org'}]
            recipe.write_text(json.dumps(config), encoding='utf-8')
            self.assertEqual(build(recipe, root/'unrelated')['accepted_provisional'], 1)

    def test_hash_bound_assistant_review_does_not_make_gold_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, recipe = self.fixture(root, [self.row('a')])
            review = root/'review.json'
            decision = {'doc_id': 'a', 'text_sha256': digest(TEXT), 'keep': True,
                        'reason': 'Text inspected', 'reviewed_extent': 'entire_candidate_text'}
            review.write_text(json.dumps({'reviewer_kind':'assistant', 'decisions':[decision]}), encoding='utf-8')
            config = json.loads(recipe.read_text()); config['inputs'][0]['require_review'] = True
            config['require_review_filters'] = [{'source_domain':'example.org'}]
            config['reviews'] = [{'path':str(review), 'sha256':file_hash(review)}]
            recipe.write_text(json.dumps(config), encoding='utf-8')
            result = build(recipe, root/'out')
            self.assertEqual(result['accepted_provisional'], 1)
            self.assertFalse(result['expert_verified'])
            review.write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError): build(recipe, root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_failed_html_revision_cannot_fall_back_into_main_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); snapshot = root/'page.html.gz'
            raw = '<div class="article-prose"><p>Хабар.</p><p><a href="/article/2-other">Басқа ұзақ мақаланың қызықты тақырыбы</a></p></div>'
            snapshot.write_bytes(gzip.compress(raw.encode()))
            row = self.row('a', source_domain='egemen.kz', source_url='https://egemen.kz/article/1-current',
                           source_extraction_status='article_prose_html', source_snapshot=str(snapshot),
                           source_sha256=file_hash(snapshot), snapshot_compression='gzip')
            _, recipe = self.fixture(root, [row]); result = build(recipe, root/'out')
            self.assertEqual((result['accepted_provisional'], result['review']), (0, 1))
            self.assertEqual(result['review_reasons']['html_extraction_failed'], 1)
            snapshot.write_bytes(b'bad')
            with self.assertRaises(ValueError): build(recipe, root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_different_recommendations_do_not_hide_duplicate_article_bodies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows = []
            for key in ['a', 'b']:
                page = root/(key+'.html.gz')
                page.write_bytes(gzip.compress((f'<div class="article-prose"><p>{TEXT}</p>'
                    f'<p><a href="/article/other-{key}">Басқа жаңалық {key}</a></p></div>').encode()))
                rows.append(self.row(key, text=TEXT+' '+key, source_domain='egemen.kz',
                    source_url='https://egemen.kz/article/'+key, source_extraction_status='article_prose_html',
                    source_snapshot=str(page), source_sha256=file_hash(page), snapshot_compression='gzip'))
            _, recipe = self.fixture(root, rows); result = build(recipe, root/'out')
            self.assertEqual(result['accepted_provisional'], 1)
            self.assertEqual(result['excluded_reasons']['normalized_exact_duplicate'], 1)

    def test_shared_passages_are_grouped_without_silently_deleting_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, recipe = self.fixture(root, [self.row('a'), self.row('b', style='literary', text=TEXT+' Балалар қуанып үйлеріне қайтты.')], passage_grouping=True)
            result = build(recipe, root/'out')
            self.assertEqual(result['accepted_provisional'], 2)
            self.assertEqual(result['grouping']['split_groups'], 1)
            self.assertEqual(result['grouping']['mixed_style_groups'], 1)
            meta = [json.loads(l) for l in (root/'out/metadata.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(meta[0]['split_group_id'], meta[1]['split_group_id'])


if __name__ == '__main__':
    unittest.main()
