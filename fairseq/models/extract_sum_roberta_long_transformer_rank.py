
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils

from fairseq.modules import (
    LearnedPositionalEmbedding, MultiheadAttention,
    SinusoidalPositionalEmbedding,
)
from fairseq.modules.attn_score_bert_encoder import AttnScoreBertEncoder
from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqModel,
    register_model, register_model_architecture, FairseqDecoder
)

from pytorch_transformers.modeling_bert import BertEncoder, BertLayerNorm

def get_sent_end_repr(src_emb, sent_ends):
    bsz, nsent = sent_ends.size()
    assert bsz == src_emb.size(0)
    seqlen = src_emb.size(1)
    offset = torch.linspace(0, (bsz-1)*seqlen, bsz).type(sent_ends.type())
    sent_ends_abs = sent_ends + offset.view(-1, 1)
    sent_ends_repr = src_emb.contiguous().view(bsz*seqlen, -1)[sent_ends_abs]
    sent_ends_repr = sent_ends_repr.view(bsz, nsent, -1)

    return sent_ends_repr


@register_model('extract_sum_roberta_long_transformer_rank')
class ExtractSumRobertaLongTransformerModel(FairseqModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', default=False, action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', default=False, action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', default=False, action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', default=False, action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', default=False, action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--roberta-model', default='roberta-base', help="RoBERTa pre-trained model selected in the list: roberta-base, "
                                "roberta-large")
        parser.add_argument('--sentence-transformer-arch', default='fairseq', help="sentence level transformer architecture [fairseq, bert]")
        parser.add_argument('--bert-no-decay', default=False, action='store_true', help="no decay for bias and LayerNorm.weight")
        parser.add_argument('--attn-type', default='attn_score', choices=['attn_score', 'attn_prob'], help='attn_socre is before softmax and attn_prob is after softmax')
        parser.add_argument('--transpose_method', choices=['transpose', 'not_transpose', 'add'], default='add')
        parser.add_argument('--attn-layer', default='all', choices=['all', 'final'])  
        parser.add_argument('--topk', default=3, type=int)


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise RuntimeError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = RankDecoder(args, tgt_dict)
        return ExtractSumRobertaLongTransformerModel(args, encoder, decoder)

    def forward(self, src_tokens, segment_ids, doc_pad_mask, doc_pos_tok, cls_pos, token_mask):
        encoder_out = self.encoder(src_tokens, segment_ids, doc_pad_mask, doc_pos_tok, cls_pos, token_mask)
        decoder_out = self.decoder(self.args.transpose_method,encoder_out)
        return decoder_out


class TransformerEncoder(FairseqEncoder):
    """Transformer encoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout

        # from pytorch_transformers import RobertaModel
        from fairseq.modules.roberta_causal_mask import RobertaCasulMaskModel
        from pytorch_transformers.file_utils import PYTORCH_TRANSFORMERS_CACHE

        self.roberta = RobertaCasulMaskModel.from_pretrained(args.roberta_model,
                cache_dir=PYTORCH_TRANSFORMERS_CACHE / 'distributed_{}'.format(args.distributed_rank))
        self.roberta.pooler.dense.weight.requires_grad = False
        self.roberta.pooler.dense.bias.requires_grad = False

        embed_dim = embed_tokens.embedding_dim

        # self.embed_tokens = embed_tokens
        # self.embed_scale = math.sqrt(embed_dim)

        self.args = args

        if args.sentence_transformer_arch == 'fairseq':
            self.padding_idx = embed_tokens.padding_idx

            self.sent_embed_positions = PositionalEmbedding(
                1024, embed_dim, self.padding_idx,
                left_pad=False,
                learned=args.encoder_learned_pos,
            )

            self.doc_layers = nn.ModuleList([])
            self.doc_layers.extend([
                TransformerEncoderLayer(args)
                for i in range(args.encoder_layers)
            ])
        elif args.sentence_transformer_arch == 'bert':
            from pytorch_transformers import RobertaConfig, RobertaTokenizer

            self.config = RobertaConfig.from_pretrained(args.roberta_model)
            self.config.output_attentions = True
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

            embed_dim = self.config.hidden_size
            print('*** padding idx before ***', embed_tokens.padding_idx)
            self.padding_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
            print('*** padding idx after ***', self.padding_idx)

            # let's assume each document has at most 128-self.padding_idx-1 sentences
            # in case of roberta, it is 126
            self.sent_position_embeddings = nn.Embedding(128, embed_dim)
            if args.encoder_layers:
                self.config.num_hidden_layers = args.encoder_layers
            if args.dropout:
                self.config.hidden_dropout_prob = args.dropout
            if args.attention_dropout:
                self.config.attention_probs_dropout_prob = args.attention_dropout
            if args.attn_type == 'attn_score':
                self.sent_encoder = AttnScoreBertEncoder(self.config)
            elif args.attn_type == 'attn_prob':
                self.sent_encoder = BertEncoder(self.config)
            else:
                raise Exception('--attn-type doesn\'t support {} yet !'.format(args.attn_type))
            self.sent_encoder.apply(self._init_weights)

            print('*** sentence encoder config ***')
            print(self.config)
        else:
            raise Exception('--sentence-transformer-arch doesn\'t support {} yet!'.format(args.sentence_transformer_arch))

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, src_tokens, segment_ids, doc_pad_mask, doc_pos_tok, cls_pos, attention_mask):
        if self.args.sentence_transformer_arch == 'fairseq':
            bsz, seqlen = src_tokens.size()

            # compute padding mask
            attention_mask = src_tokens.ne(self.padding_idx)
            # enc_hids, _ = self.bert(src_tokens, segment_ids, attention_mask, output_all_encoded_layers=False)
            all_hids = self.roberta(src_tokens, segment_ids, attention_mask=None)
            # print('all_hids', all_hids.size())
            enc_hids = all_hids[0]
            doc_pos = self.sent_embed_positions(doc_pos_tok)

            sent_repr = get_sent_end_repr(enc_hids, cls_pos)

            sent_repr = sent_repr + doc_pos
            # n_sent x bsz x C
            sent_repr = sent_repr.transpose(0, 1)
            for doc_layer in self.doc_layers:
                sent_repr = doc_layer(sent_repr, doc_pad_mask)

            return {
                'encoder_out': sent_repr,  # n_sent x bsz x C
                'encoder_padding_mask': doc_pad_mask,  # bsz x n_sent
            }
        elif self.args.sentence_transformer_arch == 'bert':
            bsz, seqlen = src_tokens.size()

            doclen = cls_pos.size(1)
            position_ids = torch.arange(1+self.padding_idx, doclen+1+self.padding_idx, dtype=torch.long, device=cls_pos.device)
            position_ids = position_ids.unsqueeze(0).expand_as(cls_pos)
            doc_pos = self.sent_position_embeddings(position_ids)

            # compute padding mask
            if attention_mask is None:
                attention_mask = src_tokens.ne(self.padding_idx)
            all_hids = self.roberta(src_tokens, segment_ids, attention_mask)
            enc_hids = all_hids[0]

            sent_repr = get_sent_end_repr(enc_hids, cls_pos)

            sent_repr = sent_repr + doc_pos

            head_mask = [None] * self.config.num_hidden_layers

            extended_doc_mask = doc_pad_mask.unsqueeze(1).unsqueeze(2)
            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            extended_doc_mask = extended_doc_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_doc_mask = extended_doc_mask * -10000.0

            all_hids_doc = self.sent_encoder(sent_repr, extended_doc_mask, head_mask)
            sent_repr_given_doc = all_hids_doc[0]
            attn_weights = all_hids_doc[1]
            return {
                'encoder_out': sent_repr_given_doc,  # bsz x n_sent x C
                'attn_weights': attn_weights,
                'encoder_doc_mask': doc_pad_mask,  # bsz x n_sent
            }
        else:
            raise Exception('--sentence-transformer-arch doesn\'t support {} yet!'.format(args.sentence_transformer_arch))

    def reorder_encoder_out(self, encoder_out_dict, new_order):
        if encoder_out_dict['encoder_out'] is not None:
            encoder_out_dict['encoder_out'] = \
                encoder_out_dict['encoder_out'].index_select(1, new_order)
        if encoder_out_dict['encoder_padding_mask'] is not None:
            encoder_out_dict['encoder_padding_mask'] = \
                encoder_out_dict['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out_dict

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        # return self.embed_positions.max_positions()
        return 10240

    def upgrade_state_dict(self, state_dict):
        '''
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'encoder.embed_positions.weights' in state_dict:
                del state_dict['encoder.embed_positions.weights']
            if 'encoder.embed_positions._float_tensor' not in state_dict:
                state_dict['encoder.embed_positions._float_tensor'] = torch.FloatTensor()
        '''
        return state_dict


class RankDecoder(FairseqDecoder):
    """Transformer decoder."""

    def __init__(self, args, tgt_dic):
        super().__init__(tgt_dic)
        self.dropout = args.dropout
        self.topk = args.topk if args.topk else 3
        self.args = args

    def forward(self, transpose_method, encoder_out):
        x = encoder_out['encoder_out']
        attn_weights = encoder_out['attn_weights']
    
        attn_head_avg = [weight.mean(dim=1) for weight in attn_weights]
        if self.args.attn_layer == 'all':
            attn_weights = sum(attn_head_avg)
        elif self.args.attn_layer == 'final':
            attn_weights = attn_head_avg[-1] 
        else:
            raise RuntimeError     
        attn_weights = F.relu(attn_weights)
        mask = encoder_out['encoder_doc_mask']

        bsz, n_sent = mask.shape

        attn_weights = F.relu(attn_weights)

        if transpose_method == 'transpose':
            attn_weights = attn_weights.transpose(1, 2)
            # print('transpose')
        elif transpose_method == 'add':
            attn_weights += attn_weights.transpose(1, 2)
            # print('add')
        elif transpose_method == 'not_transpose':
            # print('not transpose')
            pass
        else:
            raise RuntimeError('error transpose_method')

        # weights = torch.stack([torch.triu(weight) * lam_1 + torch.tril(weight) * lam_2  for weight in attn_weights])
        # eye = torch.stack([torch.eye(weights.shape[1], device='cuda').byte() for _ in weights])
        # weights.masked_fill_(eye, 0)        
        weights = attn_weights

        pr = weights.sum(dim=1)
        # pr = torch.stack([weight.diag() for weight in weights])

        out = torch.ones([bsz, n_sent], dtype=torch.int32).cuda() * self.dictionary.indices['F']
        pad = torch.ones_like(out).int() * self.dictionary.indices['<pad>']
        false = torch.ones_like(out) * self.dictionary.indices['F']

        topk = pr.argsort(dim=-1)[:, -self.topk:]
        # topk = torch.stack([torch.tensor([0, 1, 2]).type_as(t) for t in topk])

        for idx in range(bsz):
            out[idx, topk[idx]] = self.dictionary.indices['T']
        # out = false
        out = torch.where(mask==0, out, pad).int()
        return out, pr

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return 1024

    def upgrade_state_dict(self, state_dict):
        '''
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'decoder.embed_positions.weights' in state_dict:
                del state_dict['decoder.embed_positions.weights']
            if 'decoder.embed_positions._float_tensor' not in state_dict:
                state_dict['decoder.embed_positions._float_tensor'] = torch.FloatTensor()
        '''
        return state_dict


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: dropout -> add residual -> layernorm.
    In the tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    dropout -> add residual.
    We default to the approach in the paper, but the tensor2tensor approach can
    be enabled by setting `normalize_before=True`.
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
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


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings)
    return m


@register_model_architecture('extract_sum_roberta_long_transformer_rank', 'extract_sum_roberta_long_transformer_rank')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)

# Medium size transformer
@register_model_architecture('extract_sum_roberta_long_transformer_rank', 'extract_sum_roberta_long_transformer_rank_medium')
def transformer_medium(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    args.dropout = getattr(args, 'dropout', 0.1)
    base_architecture(args)


# Medium size transformer
@register_model_architecture('extract_sum_roberta_long_transformer_rank', 'extract_sum_roberta_long_transformer_rank_large')
def transformer_medium(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4086)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4086)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.1)
    base_architecture(args)
