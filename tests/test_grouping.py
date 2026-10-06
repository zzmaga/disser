import unittest
import pandas as pd
from kazstyle.data.grouping import assign_split_groups, validate_group_separation
from kazstyle.data.corpus import balanced_splits


class ProvenanceGroupingTests(unittest.TestCase):
    def test_author_and_parent_links_are_transitive_without_unknown_author_merging(self):
        frame = pd.DataFrame([
            {'doc_id':'a','parent_group':'p1','author_group':'writer'},
            {'doc_id':'b','parent_group':'p2','author_group':'writer'},
            {'doc_id':'c','parent_group':'p2','author_group':None},
            {'doc_id':'d','parent_group':'p3','author_group':None},
            {'doc_id':'e','parent_group':'p4','author_group':float('nan')}])
        groups = assign_split_groups(frame)
        self.assertEqual(len(set(groups[:3])), 1)
        self.assertEqual(len(set(groups)), 3)
        frame['split'] = ['train','validation','validation','train','test']
        with self.assertRaisesRegex(ValueError, 'author_group'):
            validate_group_separation(frame)

    def test_one_author_in_two_styles_is_not_silently_quarantined_as_bad_labels(self):
        frame = pd.DataFrame([{'doc_id':'a','author_group':'writer','label':0},
                              {'doc_id':'b','author_group':'writer','label':1}])
        frame['split_group_id'] = assign_split_groups(frame)
        with self.assertRaisesRegex(ValueError, 'Mixed-label split_group_id'):
            balanced_splits(frame, group_key='split_group_id')

    def test_reordering_rows_preserves_group_identifiers(self):
        frame = pd.DataFrame([{'doc_id':'a','template_family_id':'forms'},
                              {'doc_id':'b','template_family_id':'forms'}, {'doc_id':'c'}])
        expected = dict(zip(frame.doc_id, assign_split_groups(frame)))
        reversed_frame = frame.iloc[::-1].reset_index(drop=True)
        actual = dict(zip(reversed_frame.doc_id, assign_split_groups(reversed_frame)))
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
