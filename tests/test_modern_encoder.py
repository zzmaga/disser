"""Offline architecture compatibility; random tiny weights are not research results."""
import io
import unittest

import torch
from transformers import ModernBertConfig
from kazstyle.models.style_transformer import StyleTransformer


class ModernEncoderTests(unittest.TestCase):
    def test_cpu_forward_backward_and_weight_restoration_without_downloads(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        config=ModernBertConfig(vocab_size=128,hidden_size=24,intermediate_size=48,
            num_hidden_layers=4,num_attention_heads=4,max_position_embeddings=64,
            local_attention=16,pad_token_id=0,bos_token_id=1,eos_token_id=2,
            cls_token_id=1,sep_token_id=2)
        model=StyleTransformer('offline_tiny_modernbert',5,encoder_config=config).eval()
        tokens=torch.tensor([[1,30,40,2,0],[1,55,56,57,2]])
        inputs={'input_ids':tokens,'attention_mask':tokens.ne(0).long()}
        scores=model(**inputs)
        self.assertEqual(tuple(scores.shape),(2,5));self.assertTrue(torch.isfinite(scores).all())
        torch.nn.functional.cross_entropy(scores,torch.tensor([0,4])).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters()))
        buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
        restored=StyleTransformer('offline_tiny_modernbert',5,encoder_config=config).eval()
        restored.load_state_dict(torch.load(buffer,weights_only=True,map_location='cpu'))
        with torch.no_grad():
            self.assertTrue(torch.allclose(scores.detach(),restored(**inputs),atol=1e-6,rtol=1e-6))


if __name__=='__main__':unittest.main()
