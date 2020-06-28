from pytorch_transformers.modeling_bert import (
    BertEncoder, BertLayer, BertAttention, BertSelfAttention,
)
import math
import torch
from torch import nn


class AttnScoreBertEncoder(BertEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([AttnScoreBertLayer(config) for _ in range(config.num_hidden_layers)])


class AttnScoreBertLayer(BertLayer):
    
    def __init__(self, config):
        super().__init__(config)
        self.attention = AttnScoreBertAttention(config)


class AttnScoreBertAttention(BertAttention):
    
    def __init__(self, config):
        super().__init__(config)
        self.self = AttnScoreBertSelfAttention(config)


class AttnScoreBertSelfAttention(BertSelfAttention):
    
    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs
