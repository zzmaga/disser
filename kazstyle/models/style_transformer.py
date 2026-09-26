"""Explicitly named pretrained encoder with selectable pooling heads."""
import torch
from torch import nn
from transformers import AutoModel


class StyleTransformer(nn.Module):
    def __init__(self, model_name, num_labels, head='last', local_only=True, encoder_config=None):
        super().__init__()
        if head not in {'last','concat4'}:
            raise ValueError('head must be last or concat4')
        self.encoder = (AutoModel.from_config(encoder_config) if encoder_config is not None
                        else AutoModel.from_pretrained(model_name,local_files_only=local_only))
        self.head = head
        if head=='concat4' and self.encoder.config.num_hidden_layers<4:
            raise ValueError('concat4 requires at least four encoder layers')
        width = self.encoder.config.hidden_size*(4 if head=='concat4' else 1)
        self.dropout = nn.Dropout(.1)
        self.classifier = nn.Linear(width,num_labels)

    def forward(self,input_ids,attention_mask,**kwargs):
        result = self.encoder(input_ids=input_ids,attention_mask=attention_mask,
                              output_hidden_states=self.head=='concat4',return_dict=True)
        pooled = (torch.cat([x[:,0,:] for x in result.hidden_states[-4:]],dim=-1)
                  if self.head=='concat4' else result.last_hidden_state[:,0,:])
        return self.classifier(self.dropout(pooled))
