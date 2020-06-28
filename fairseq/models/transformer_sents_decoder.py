import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from fairseq import utils

from fairseq.modules import (
    LearnedPositionalEmbedding, MultiheadAttention,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqEncoderDecoderModel,
    register_model, register_model_architecture,
)

def get_sent_end_repr(src_emb, sent_ends):
    bsz, nsent = sent_ends.size()
    assert bsz == src_emb.size(0)
    seqlen = src_emb.size(1)
    offset = torch.linspace(0, (bsz-1)*seqlen, bsz).type(sent_ends.type())
    sent_ends_abs = sent_ends + offset.view(-1, 1)
    sent_ends_repr = src_emb.contiguous().view(bsz*seqlen, -1)[sent_ends_abs]
    sent_ends_repr = sent_ends_repr.view(bsz, nsent, -1)

    return sent_ends_repr

class TransformerSentDecoder(FairseqIncrementalDecoder):
    """Transformer decoder."""

    def __init__(self, args, src_dictionary, dictionary, embed_tokens, left_pad=False):
        '''
        The decoder has two parts:
            1) a transformer decoder to predict masked sentences
            2) a decoder to predict sentence labels
        '''
        super().__init__(src_dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        embed_dim = embed_tokens.embedding_dim
        padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim, padding_idx,
            learned=args.decoder_learned_pos,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args)
            for i in range(args.decoder_layers)
        ])

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(src_dictionary), embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)

        # print('embed_out size', self.embed_out.size())

        # this part is for sentence label prediction
        # self.out_proj = Linear(embed_dim, len(dictionary))
        self.sentence_transformer_arch = args.sentence_transformer_arch

    # def forward_sent_label(self, x):
    #     x = F.dropout(x, p=self.dropout, training=self.training)
    #     return self.out_proj(x)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, '_future_mask')
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def forward_masked_sent(self, encoder_sent_repr, masked_sent_positions, prev_output_tokens, incremental_state=None):
        # encoder_sent_repr: bsz x n_sent x C
        # masked_sent_positions: bsz x max_num_mask
        # masked_encoder_sent_repr: bsz x max_num_mask x C
        masked_encoder_sent_repr = get_sent_end_repr(encoder_sent_repr, masked_sent_positions)
        masked_encoder_sent_repr_2d = masked_encoder_sent_repr.view(
                                        masked_encoder_sent_repr.size(0)*masked_encoder_sent_repr.size(1),
                                        masked_encoder_sent_repr.size(2) )

        # prev_output_tokens: bsz x max_n_sent x T --> (bsz x max_n_sent) x T
        prev_output_tokens = prev_output_tokens.view(prev_output_tokens.size(0)*prev_output_tokens.size(1),
                             prev_output_tokens.size(2))

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        self_attn_mask = self.buffered_future_mask(x)
        # decoder layers
        for layer in self.layers:
            x = layer(
                x,
                masked_encoder_sent_repr_2d,
                incremental_state,
                self_attn_mask,
            )

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # project back to size of vocabulary
        if self.share_input_output_embed:
            x = F.linear(x, self.embed_tokens.weight)
        else:
            x = F.linear(x, self.embed_out)

        return x

    def forward(self, encoder_out, masked_sent_positions, prev_output_tokens, incremental_state=None):
        # n_sent x bsz x C
        x = encoder_out['encoder_out']
        # n_sent x bsz x C -> bsz x n_sent x C
        if self.sentence_transformer_arch == 'fairseq':
            x = x.transpose(0, 1)

        # predict sentence label
        # sent_label = self.forward_sent_label(x)

        # predict masked sentence
        masked_sent = self.forward_masked_sent(x, masked_sent_positions, prev_output_tokens, incremental_state)

        return masked_sent

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return 1024

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'decoder.embed_positions.weights' in state_dict:
                del state_dict['decoder.embed_positions.weights']
            # if 'decoder.embed_positions._float_tensor' in state_dict:
            #     del state_dict['decoder.embed_positions._float_tensor']
            state_dict['decoder_perm.embed_positions._float_tensor'] = torch.FloatTensor(1)
        
        '''
        in_proj_weight -> q_proj.weight, k_proj.weight, v_proj.weight
        in_proj_bias -> q_proj.bias, k_proj.bias, v_proj.bias
        '''
        def transform_params(idx, suffix):
            in_proj_ = state_dict['decoder.layers.{}.self_attn.in_proj_{}'.format(idx, suffix)]
            del state_dict['decoder.layers.{}.self_attn.in_proj_{}'.format(idx, suffix)]
            state_dict['decoder.layers.{}.self_attn.q_proj.{}'.format(idx, suffix)], state_dict['decoder.layers.{}.self_attn.k_proj.{}'.format(idx, suffix)],\
            state_dict['decoder.layers.{}.self_attn.v_proj.{}'.format(idx, suffix)] = in_proj_.chunk(3, dim=0)

        if 'decoder.layers.0.self_attn.in_proj_weight' in state_dict:
            for idx in range(len(self.layers)):
                transform_params(idx, 'weight')

        if 'decoder.layers.0.self_attn.in_proj_bias' in state_dict:
            for idx in range(len(self.layers)):
                transform_params(idx, 'bias')

        return state_dict


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before
        '''
        self.encoder_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        '''
        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(3)])

    def forward(self, x, encoder_out, incremental_state, self_attn_mask=None):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=self_attn_mask,
            incremental_state=incremental_state,
            need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        # add encoder information
        residual = x # T x bsz x C
        # print('residual x', residual.size())
        residual = self.maybe_layer_norm(1, residual, before=True)
        # encoder_out: only one per sequence
        # bsz x C --> 1 x bsz x C
        x = encoder_out.view(1, encoder_out.size(0), encoder_out.size(1))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        # add encoder information done!

        residual = x
        x = self.maybe_layer_norm(2, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(2, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m