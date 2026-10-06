import unittest
from unittest.mock import patch
import pandas as pd
from kazstyle.data.training_views import shorten


class WordTokenizer:
    def __call__(self,text,**kwargs):return {'input_ids':list(range(len(text.split())+2))}


class TrainingViewTests(unittest.TestCase):
    def test_new_containment_reverts_train_only_and_preserves_evaluation(self):
        def block(prefix,n):return [prefix+str(i) for i in range(n)]
        common=block('common',25)
        texts=[' '.join(block('left',70)+common+block('right',65)),
               ' '.join(block('validationleft',70)+common+block('validationright',65)),
               ' '.join(block('independent',160))]
        frame=pd.DataFrame([{'doc_id':str(i),'sample_id':str(i),'excerpt_hash':str(i),
            'text':text,'split':'validation' if i==1 else 'train','excerpt_words':160,'token_count':162,'view_max_words':160}
            for i,text in enumerate(texts)])
        with patch('kazstyle.data.training_views.inspect_text',return_value=({},[])):
            result,audit=shorten(frame,WordTokenizer(),{'max_tokens':256},[40])
        self.assertEqual(result.loc[0,'text'],texts[0])
        self.assertTrue(result.loc[[1]].equals(frame.loc[[1]]))
        self.assertEqual(result.loc[2,'excerpt_words'],40)
        self.assertTrue(next(r for r in audit if r['doc_id']=='0')['reverted_for_cross_split_passage'])
        self.assertEqual(len(result),len(frame))


if __name__=='__main__':unittest.main()
