from pytorch_transformers import RobertaModel, BertModel
import torch

class AbstractCasulMaskModel(object):

# class RobertaCasulMaskModel(RobertaModel):
    def new_forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
    
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if len(attention_mask.shape) == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif len(attention_mask.shape) == 4:
            correct_shape = torch.Size([input_ids.shape[0], 1, input_ids.shape[-1], input_ids.shape[-1]])
            assert attention_mask.shape == correct_shape, 'please check the mask which has a wrong shape.'
            extended_attention_mask = attention_mask
        else:
            raise RuntimeError('please check the mask which has a wrong shape.')

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

    def expand_position_embedding(self, new_max_position, initializer_range):
        num_expand = new_max_position - 512
        old_position, embed_dim = self.embeddings.position_embeddings.weight.shape

        new_embeddings = torch.nn.Embedding(old_position + num_expand, embed_dim)
        new_embeddings.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        new_embeddings.to(self.embeddings.position_embeddings.weight.device)
        new_embeddings.to(self.embeddings.position_embeddings.weight.dtype)

        new_embeddings.weight.data[:old_position, :] = self.embeddings.position_embeddings.weight.data
        self.embeddings.position_embeddings = new_embeddings

class RobertaCasulMaskModel(RobertaModel, AbstractCasulMaskModel):

    def forward(self, input_ids, *inputs, **kwargs):
        if input_ids[:, 0].sum().item() != 0:
            logger.warning("A sequence with no special tokens has been passed to the RoBERTa model. "
                    "This model requires special tokens in order to work. "
                    "Please specify add_special_tokens=True in your encoding.")

        return self.new_forward(input_ids, *inputs, **kwargs)


class BertCasulMaskModel(BertModel, AbstractCasulMaskModel):

    def forward(self, *inputs, **kwargs):

        return self.new_forward(*inputs, **kwargs)