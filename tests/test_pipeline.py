import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from kazstyle.data.corpus import (balanced_splits,canonical_url,duplicate_groups,normalize_text,
                        read_csv_strict,validate_manifest,make_excerpt)


class CorpusTests(unittest.TestCase):
    def test_csv_quoted_commas_newlines_and_long_fields(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'data.csv'
            with path.open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['text','label','source_url'])
                writer.writerow(['Абай, "өлең"\nекінші жол','literary','https://a.kz/1'])
                writer.writerow(['қазақ '*50000,'literary','https://a.kz/2'])
            rows,audit=read_csv_strict(path)
            self.assertEqual(len(rows),2)
            self.assertIn('\n',rows[0]['text'])
            self.assertEqual(audit['embedded_newline_records'],1)
            self.assertEqual(len(rows[1]['text'].split()),50000)

    def test_malformed_csv_fails_instead_of_dropping_record(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'bad.csv'
            path.write_text('text,label,source_url\nfoo,official,https://x,extra\n',encoding='utf-8')
            with self.assertRaises(ValueError):read_csv_strict(path)

    def test_cleaning_preserves_style_and_names(self):
        value=normalize_text('Абай! Әділет туралы 😊\nБөлісу туралы әңгіме. https://x.kz/a')
        self.assertEqual(value,'Абай! Әділет туралы 😊 Бөлісу туралы әңгіме.')
        self.assertEqual(canonical_url('https://EXAMPLE.kz/a?q=2#chunk=3'),'https://example.kz/a?q=2')

    def test_duplicate_families_are_transitive(self):
        d=pd.DataFrame({'doc_id':['a','b','c'],'content_hash':['x','x','z'],
                        'excerpt_hash':['u','v','v'],'text':['алма','алмұрт','өрік'],
                        'duplicate_probe':['алма','алмұрт','өрік']})
        groups,_=duplicate_groups(d)
        self.assertEqual(len(set(groups)),1)

    def test_balancing_does_not_leak_groups(self):
        rows=[]
        for label in range(3):
            for i in range(120):
                key=f'{label}_{i}'
                rows.append(dict(sample_id=key,doc_id=key,duplicate_group_id=f'{label}_{i//2}',
                                 excerpt_hash=key,text='test',label=label))
        frame=pd.DataFrame(rows)
        first,_=balanced_splits(frame,42,100)
        second,_=balanced_splits(frame,42,100)
        self.assertEqual(first.sample_id.tolist(),second.sample_id.tolist())
        validate_manifest(first)
        for _,part in first.groupby('split'):
            self.assertEqual(part.label.value_counts().nunique(),1)
        leaked=first.copy()
        left=leaked.index[leaked.split=='train'][0]
        right=leaked.index[leaked.split=='test'][0]
        leaked.loc[right,'duplicate_group_id']=leaked.loc[left,'duplicate_group_id']
        with self.assertRaises(ValueError):validate_manifest(leaked)

    @unittest.skipUnless(Path('data/processed/pilot_v2/tokenizer').exists(), 'Integration test requires built pilot tokenizer')
    def test_excerpt_respects_real_tokenizer_limit(self):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained('data/processed/pilot_v2/tokenizer',local_files_only=True)
        text=('Қазақстанда ғылыми зерттеулердің нәтижелері жарияланды. '*100)
        excerpt,details=make_excerpt(text,tokenizer,max_words=160,max_tokens=64)
        self.assertTrue(excerpt)
        self.assertLessEqual(len(tokenizer(excerpt)['input_ids']),64)
        self.assertEqual(details['token_count'],len(tokenizer(excerpt)['input_ids']))
        self.assertGreater(details['tokens_before_limit'],64)

    def test_external_import_budget_keeps_only_complete_rows(self):
        from kazstyle.data.import_hf import bounded_rows,DownloadBudgetExceeded
        def source():
            yield {'text':'complete'}
            raise DownloadBudgetExceeded()
        state={'stop_reason':'end_of_file'}
        self.assertEqual(list(bounded_rows(source(),state)),[{'text':'complete'}])
        self.assertEqual(state['stop_reason'],'byte_budget')


if __name__=='__main__':unittest.main()
