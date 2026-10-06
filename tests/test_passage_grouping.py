import unittest
import pandas as pd
from kazstyle.data.grouping import assign_passage_groups, assign_split_groups, validate_group_separation


class PassageGroupingTests(unittest.TestCase):
    def test_shared_passage_links_different_documents_transitively(self):
        a=' '.join('alpha'+str(i) for i in range(60))
        b=' '.join('beta'+str(i) for i in range(60))
        frame=pd.DataFrame([
            {'doc_id':'a','text':a+' additional first ending'},
            {'doc_id':'b','text':a+' '+b},
            {'doc_id':'c','text':b+' distinct ending'},
            {'doc_id':'d','text':'independent small document unrelated to the other passages'}])
        frame['passage_group_id'],edges=assign_passage_groups(frame)
        self.assertEqual(frame.passage_group_id.iloc[0],frame.passage_group_id.iloc[2])
        self.assertTrue(pd.isna(frame.passage_group_id.iloc[3]))
        self.assertEqual(len(edges),2)
        frame['split_group_id']=assign_split_groups(frame)
        frame['split']=['train','train','test','test']
        with self.assertRaisesRegex(ValueError,'passage_group_id'):
            validate_group_separation(frame)

    def test_short_exact_text_and_punctuation_are_grouped(self):
        f=pd.DataFrame({'doc_id':['a','b','c'],'text':['ONE, two!','one two','unrelated words']})
        groups,edges=assign_passage_groups(f)
        self.assertEqual(groups[0],groups[1]); self.assertIsNone(groups[2])
        self.assertEqual(edges[0]['flag'],'normalized_exact')


if __name__=='__main__':unittest.main()
