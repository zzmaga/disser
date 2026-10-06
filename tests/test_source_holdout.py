import unittest
import pandas as pd
from kazstyle.data.source_holdout import allocate


class SourceHoldoutTests(unittest.TestCase):
    def test_entire_cross_publisher_family_is_excluded_from_training(self):
        rows=[]
        for label,style in enumerate(['official','scientific']):
            for i in range(12):
                key=f'{style}_{i}'
                rows.append({'sample_id':key,'doc_id':key,'duplicate_group_id':key,
                    'excerpt_hash':key,'text':key,'label':label,'style':style,
                    'split':'train','split_group_id':f'{style}_shared' if i<2 else key,
                    'source_domain':f'held_{style}' if i in [0,2,3] else 'train_domain'})
        recipe={'heldout_sources':{s:[f'held_{s}'] for s in ['official','scientific']},
                'train_per_style':4,'validation_per_style':2,'test_per_style':2,'seed':42}
        result=allocate(pd.DataFrame(rows),recipe)
        self.assertEqual(result.groupby('split').size().to_dict(),{'test':4,'train':8,'validation':4})
        self.assertFalse(result[result.split!='test'].doc_id.str.endswith('_1').any())
        self.assertEqual(set(result[result.split!='test'].source_domain),{'train_domain'})


if __name__=='__main__':unittest.main()
