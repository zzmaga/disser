import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kazstyle.data.corpus import digest, file_hash, write_json
from kazstyle.data.reservation import (allocate, assert_training_allowed, identities,
                                      protected_rows, registered_ledgers, select_candidates)


def row(name, group=None, domain='train.example', style='literary', **extra):
    return {'doc_id': name, 'style': style, 'source_domain': domain,
            'source_url': f'https://{domain}/{name}', 'split_group_id': group or name,
            'text_sha256': digest(name), 'quality': {'words': 40}, 'genre': 'prose', **extra}


class ReservationTests(unittest.TestCase):
    def test_source_and_document_reservations_close_over_unselected_bridge_members(self):
        rows = [row('source', 's', 'withheld.example'), row('source_bridge', 's'),
                row('new', 'n'), row('bridge', 'n'), row('known', 'k'), row('old_alias', 'k'),
                row('other', 'other')]
        policy = {'styles': ['literary'], 'source_selectors': {'literary': {'source_domain': 'withheld.example'}},
                  'seed': 42, 'candidates_per_style_per_role': 1}
        chosen, protected, eligible, blocked = allocate(rows, [row('known')], policy)
        self.assertEqual(blocked, {'k'})
        self.assertFalse(set(r['doc_id'] for r in chosen) & {'known', 'old_alias'})
        self.assertTrue({'source', 'source_bridge'} <= {r['doc_id'] for r in protected})
        forbidden = {r['split_group_id'] for r in protected}
        self.assertFalse(forbidden & {r['split_group_id'] for r in eligible})
        self.assertEqual(len(chosen), 2)

    def test_archive_url_is_not_customer_review_identity(self):
        one = row('one', domain='github.com', source_family='KazSAnDRA', upstream_domain='bookstore',
                  source_url='https://raw.githubusercontent.com/example/dataset/archive.zip')
        two = row('two', domain='github.com', source_family='KazSAnDRA', upstream_domain='market')
        two['source_url'] = one['source_url']
        self.assertNotIn(('source_url', one['source_url']), identities(one))
        ledger = {'reservation_id': 'test', 'source_selectors': [
            {'source_family': 'KazSAnDRA', 'upstream_domain': 'bookstore'}],
            'protected_identities': {'doc_id': ['one']}}
        self.assertEqual(set(protected_rows([one, two], [(None, ledger)])), {0})

    def test_dataset_landing_page_does_not_merge_legal_registry_documents(self):
        one=row('legal1',domain='zan.gov.kz',source_url='https://huggingface.co/datasets/example/legal')
        two=row('legal2',domain='zan.gov.kz',source_url=one['source_url'])
        self.assertFalse(identities(one).intersection(identities(two)))

    def test_group_and_duplicate_provenance_block_renamed_excerpt(self):
        ledger = {'reservation_id': 'test', 'source_selectors': [],
                  'protected_identities': {'split_group_id': ['full_family'], 'doc_id': ['reserved_original']}}
        rows = [row('new_excerpt', split_group_id=None, corpus_split_group_id='full_family'),
                row('renamed', duplicate_occurrences=[{'doc_id': 'reserved_original'}]), row('allowed')]
        self.assertEqual(set(protected_rows(rows, [(None, ledger)])), {0, 1})

    def test_hash_still_blocks_metadata_stripped_exact_text(self):
        ledger = {'reservation_id': 'test', 'source_selectors': [],
                  'protected_identities': {'text_hash': [digest('unchanged original text')]}}
        self.assertEqual(set(protected_rows([{'text': 'unchanged original text'}], [(None, ledger)])), {0})

    def test_new_component_and_transitive_aliases_inherit_reservation(self):
        ledger={'reservation_id':'test','source_selectors':[],
                'protected_identities':{'doc_id':['old_reserved']}}
        rows=[row('old_reserved',group='renamed_canonical',author_group='writer'),
              row('bridge',author_group='writer',work_group='new_work'),
              row('new_story',work_group='new_work'),
              row('omitted_bridge_sibling',group=None,corpus_split_group_id='renamed_canonical'),
              row('allowed')]
        self.assertEqual(set(protected_rows(rows,[(None,ledger)])),{0,1,2,3})

    def test_registered_ledger_rejects_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ledger = root/'ledger.json'; registry = root/'registry.json'
            write_json(ledger, {'protected_identities': {}, 'source_selectors': []})
            write_json(registry, {'ledgers': [{'path': str(ledger), 'sha256': file_hash(ledger)}]})
            self.assertEqual(len(registered_ledgers(registry)), 1)
            write_json(ledger, {'changed': True})
            with self.assertRaisesRegex(ValueError, 'ledger changed'):
                registered_ledgers(registry)

    def test_missing_reservation_registry_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing=Path(tmp)/'missing.json'
            with patch('kazstyle.data.reservation.REGISTRY',missing):
                with self.assertRaises(FileNotFoundError):
                    registered_ledgers()

    def test_group_budget_is_binding_and_order_independent(self):
        rows = [row('huge', 'huge'), *(row(f'small_{i}', 'small') for i in range(3))]
        sizes = {'huge': 1000, 'small': 3}
        first = select_candidates(rows, 2, 42, group_sizes=sizes, max_embargo=3)
        second = select_candidates(list(reversed(rows)), 2, 42, group_sizes=sizes, max_embargo=3)
        self.assertEqual([r['doc_id'] for r in first], [r['doc_id'] for r in second])
        self.assertEqual({r['split_group_id'] for r in first}, {'small'})
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select_candidates(rows, 4, 42, group_sizes=sizes, max_embargo=3)

    def test_explicit_reserved_status_cannot_be_bypassed_by_empty_registry(self):
        with patch('kazstyle.data.reservation.registered_ledgers', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'Reserved evaluation material'):
                assert_training_allowed([{'usage_status': 'reserved_pending_review'}])

    def test_allowed_reserved_test_cannot_hide_a_related_training_row(self):
        evaluation=row('eval',author_group='writer',split='test',usage_status='reserved_pending_review')
        with patch('kazstyle.data.reservation.registered_ledgers',return_value=[]):
            assert_training_allowed([evaluation,row('independent',split='train')],allow_reserved_test=True)
            with self.assertRaisesRegex(ValueError,'Reserved evaluation material'):
                assert_training_allowed([evaluation,row('sibling',author_group='writer',split='train')],allow_reserved_test=True)


if __name__ == '__main__':
    unittest.main()
