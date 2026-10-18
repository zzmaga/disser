import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kazstyle.data.corpus import write_json
from kazstyle.models.tokenizer_reference import build


class TokenizerReferenceTests(unittest.TestCase):
    def test_unpinned_revision_is_rejected_before_download_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=root/'protocol.json'
            write_json(spec,{'primary_models':[{'model_name':'test/model','revision':'main'}]})
            with patch('kazstyle.models.tokenizer_reference.AutoConfig.from_pretrained') as loader:
                with self.assertRaisesRegex(ValueError,'pinned commit'):
                    build(spec,root/'out',allow_download=True)
                loader.assert_not_called()
            self.assertFalse((root/'out').exists())

    def test_duplicate_encoder_registration_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=root/'protocol.json'
            model={'model_name':'test/model','revision':'a'*40}
            write_json(spec,{'primary_models':[model,model]})
            with self.assertRaisesRegex(ValueError,'exactly once'):
                build(spec,root/'out')

    def test_wrong_resolved_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=root/'protocol.json'
            write_json(spec,{'primary_models':[{'model_name':'test/model','revision':'a'*40}]})
            with patch('kazstyle.models.tokenizer_reference.AutoConfig.from_pretrained',return_value=SimpleNamespace(_commit_hash='b'*40)),patch('kazstyle.models.tokenizer_reference.AutoTokenizer.from_pretrained') as tokenizer:
                with self.assertRaisesRegex(ValueError,'Resolved encoder revision'):
                    build(spec,root/'out')
                tokenizer.assert_not_called()
            self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
