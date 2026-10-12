import unittest
from kazstyle.evaluation.window_probe import aggregate_votes


class WindowVoteTests(unittest.TestCase):
    def test_votes_are_weighted_by_words_not_window_count(self):
        result=aggregate_votes([{'words':2,'style':'A'},{'words':2,'style':'A'},
                                {'words':100,'style':'B'}],['A','B'],'A')
        self.assertEqual(result['style'],'B')
        self.assertEqual(result['classified_words'],104)
        self.assertAlmostEqual(result['dominant_vote_word_fraction'],100/104)

    def test_tie_prefers_central_only_if_it_is_a_leader(self):
        votes=[{'words':10,'style':'A'},{'words':10,'style':'B'}]
        self.assertEqual(aggregate_votes(votes,['A','B','C'],'B')['style'],'B')
        self.assertEqual(aggregate_votes(votes,['A','B','C'],'C')['style'],'A')
        self.assertTrue(aggregate_votes(votes,['A','B'],'A')['tied'])

    def test_no_classifiable_words_produces_no_class(self):
        self.assertIsNone(aggregate_votes([],['A','B'],'A')['style'])
        with self.assertRaises(ValueError):
            aggregate_votes([{'words':0,'style':'A'}],['A'],'A')


if __name__=='__main__':unittest.main()
