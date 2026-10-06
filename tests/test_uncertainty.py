import unittest
import numpy as np
import pandas as pd
from kazstyle.evaluation.repeated import macro_from_matrix,paired_group_bootstrap


class UncertaintyTests(unittest.TestCase):
    def test_perfect_incorrect_and_identical_paired_predictions(self):
        frame=pd.DataFrame([{'sample_id':str(i),'label':i//4,'duplicate_group_id':str(i//2)} for i in range(8)])
        perfect={r.sample_id:r.label for r in frame.itertuples()}
        wrong={r.sample_id:1-r.label for r in frame.itertuples()}
        report=paired_group_bootstrap(frame,{'perfect':perfect,'same':perfect,'wrong':wrong},'perfect',repeats=100)
        self.assertEqual(report['groups'],4)
        self.assertEqual(report['models']['perfect']['macro_f1_95_percentile_interval'],[1.,1.])
        self.assertEqual(report['models']['same']['paired_macro_f1_difference_vs_reference_95_interval'],[0.,0.])
        self.assertEqual(report['models']['wrong']['paired_macro_f1_difference_vs_reference_95_interval'],[-1.,-1.])

    def test_group_crossing_labels_fails(self):
        frame=pd.DataFrame([{'sample_id':'a','label':0,'duplicate_group_id':'same'},
                            {'sample_id':'b','label':1,'duplicate_group_id':'same'}])
        with self.assertRaisesRegex(ValueError,'class-homogeneous'):
            paired_group_bootstrap(frame,{'x':{'a':0,'b':1}},'x',repeats=100)

    def test_empty_class_zero_division_is_zero(self):
        self.assertEqual(macro_from_matrix(np.array([[3,0],[0,0]])),.5)


if __name__=='__main__':unittest.main()
