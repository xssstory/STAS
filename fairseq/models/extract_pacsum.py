
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils

from fairseq.modules import (
    LearnedPositionalEmbedding, MultiheadAttention,
    SinusoidalPositionalEmbedding,
)

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqModel,
    register_model, register_model_architecture, FairseqDecoder,
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


@register_model('extract_pacsum')
class ExtractPacsum(FairseqModel):

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
        parser.add_argument('--lam1', default=None, type=float)
        parser.add_argument('--lam2', default=None, type=float)
        parser.add_argument('--beta', default=None, type=float)

        parser.add_argument('--pretrained-bert-model', default='roberta-base', help="RoBERTa pre-trained model selected in the list: roberta-base, "
                                "roberta-large", choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
        # parser.add_argument('--sentence-transformer-arch', default='fairseq', help="sentence level transformer architecture [fairseq, bert]")
        parser.add_argument('--bert-no-decay', default=False, action='store_true', help="no decay for bias and LayerNorm.weight")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        src_dict = task.source_dictionary
        tgt_dict = task.target_dictionary

        encoder = TransformerEncoder(args, src_dict)
        decoder = PacsumDecoder(args, tgt_dict)
        return cls(args, encoder, decoder)

    def forward(self, src_tokens, segment_ids, doc_pad_mask, doc_pos_tok):
        encoder_out = self.encoder(src_tokens, segment_ids, doc_pad_mask, doc_pos_tok)

        decoder_out = self.decoder(encoder_out)

        return decoder_out
        


class TransformerEncoder(FairseqEncoder):
    """Transformer encoder."""

    def __init__(self, args, dictionary, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout

        from pytorch_transformers import RobertaModel, BertModel
        from pytorch_transformers.file_utils import PYTORCH_TRANSFORMERS_CACHE
        from pytorch_transformers import RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer

        if args.pretrained_bert_model.startswith('roberta'):
            self.embed = RobertaModel.from_pretrained(args.pretrained_bert_model,
                    cache_dir=PYTORCH_TRANSFORMERS_CACHE / 'distributed_{}'.format(args.distributed_rank))
            # self.context = RobertaModel.from_pretrained(args.pretrained_bert_model,
            #         cache_dir=PYTORCH_TRANSFORMERS_CACHE / 'distributed_{}'.format(args.distributed_rank))
            self.config = RobertaConfig.from_pretrained(args.pretrained_bert_model)
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

            
        else:
            self.embed = BertModel.from_pretrained(args.pretrained_bert_model,
                    cache_dir=PYTORCH_TRANSFORMERS_CACHE / 'distributed_{}'.format(args.distributed_rank))
            # self.context = BertModel.from_pretrained(args.pretrained_bert_model,
            #         cache_dir=PYTORCH_TRANSFORMERS_CACHE / 'distributed_{}'.format(args.distributed_rank))
            self.config = BertConfig.from_pretrained(args.pretrained_bert_model)

            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.padding_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

    def forward(self, src_tokens, segment_ids, doc_pad_mask, doc_pos_tok):

        bsz, doc_len, sent_len = src_tokens.size()

        # doclen = cls_pos.size(1)
        # position_ids = torch.arange(1+self.padding_idx, doclen+1+self.padding_idx, dtype=torch.long, device=cls_pos.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(cls_pos)
        # doc_pos = self.sent_position_embeddings(position_ids)

        # compute padding mask
        attention_mask = src_tokens.ne(self.padding_idx)
        # all_hids_embed = self.embed(src_tokens, segment_ids, attention_mask)
        # all_hids_context = self.context(src_tokens, segment_ids, attention_mask)

        # enc_hids_embed = all_hids_embed[1]
        # enc_hids_context = all_hids_context[1]

        enc_hids_embed = self.embed(src_tokens.view(-1, sent_len), segment_ids.view(-1, sent_len), attention_mask.view(-1, sent_len))[1]
        # enc_hids_context = self.context(src_tokens.view(-1, sent_len), segment_ids.view(-1, sent_len), attention_mask.view(-1, sent_len))[1]

        sent_repr_embed = enc_hids_embed.view(bsz, doc_len, -1)
        # sent_repr_context = enc_hids_context.view(bsz, doc_len, -1)

        return {
            'encoder_out': {
                'embed': sent_repr_embed,  # bsz x n_sent x C
                # 'context': sent_repr_context, # bsz x 
            },
            'encoder_doc_mask': doc_pad_mask,  # bsz x n_sent
        }

    # def reorder_encoder_out(self, encoder_out_dict, new_order):
    #     if encoder_out_dict['encoder_out'] is not None:
    #         encoder_out_dict['encoder_out'] = \
    #             encoder_out_dict['encoder_out'].index_select(1, new_order)
    #     if encoder_out_dict['encoder_padding_mask'] is not None:
    #         encoder_out_dict['encoder_padding_mask'] = \
    #             encoder_out_dict['encoder_padding_mask'].index_select(0, new_order)
    #     return encoder_out_dict

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

class PacsumDecoder(FairseqDecoder):

    def __init__(self, args, tgt_dic):
        super().__init__(tgt_dic)
        self.topk = 3
        self.args=args

    @torch.no_grad()
    def forward(self, encoder_out):
        embed = encoder_out['encoder_out']['embed']
        mask = encoder_out['encoder_doc_mask']
        bsz, n_sent = mask.shape
        params = []
        if self.args.lam1 is not None and self.args.lam2 is not None and self.args.beta is not None:
            params.extend([[self.args.lam1, self.args.lam2, self.args.beta], [self.args.lam2, self.args.lam1, self.args.beta]])
        else:
            for j in range(11):
                for i in range(10):
                    beta = i * 0.1
                    lam_1 = j * 0.1
                    lam_2 = 1 - lam_1
                    params.append([lam_1, lam_2, beta])
        
        scores = [torch.zeros([bsz, n_sent], dtype=torch.float32).cuda() for _ in range(len(params) )]
        outs = [torch.ones([bsz, n_sent], dtype=torch.int32).cuda() * self.dictionary.index('F') for _ in scores]
        # weights = torch.bmm(embed, embed.transpose(-1, -2))
        for i, (lam_1, lam_2, beta) in enumerate(params):
            for idx in range(bsz):
                e = embed[idx]
                m = mask[idx]

                e = e[m==0]
                w = e.matmul(e.t())
                eye = torch.eye(w.shape[-2], device=w.device).bool()
                w.masked_fill_(eye, 1)
                w = w - w.min() - beta * (w.max() - w.min())
                w = F.relu(w)
                w.masked_fill_(eye, 0)
                w = - w.triu() * lam_1 + w.tril() * lam_2

                pr = w.sum(dim=-2)
                topk = pr.argsort(dim=-1)[-self.topk:]
                scores[i][idx, :len(pr)] = pr
                outs[i][idx, topk] = self.dictionary.indices['T']
            outs[i].masked_fill_(mask!=0, self.dictionary.pad_index)
        return outs,  scores

    '''
    def forward(self, encoder_out):
        lam_1, lam_2, beta = self.lam1, self.lam2, self.beta
        embed = encoder_out['encoder_out']['embed']
        mask = encoder_out['encoder_doc_mask']

        weights = torch.bmm(embed, embed.transpose(-1, -2))
        bsz, n_sent = mask.shape

        weights = torch.stack([weight - weight.min() - beta * (weight.max() - weight.min()) for weight in weights])
        weights = F.relu(weights)
        weights = torch.stack([torch.triu(weight) * lam_1 + torch.tril(weight) * lam_2  for weight in weights])
        eye = torch.stack([torch.eye(weights.shape[1], device='cuda').byte() for _ in weights])
        weights.masked_fill_(eye, 0)        

        pr = weights.sum(dim=1)
        # print(pr)

        out = torch.ones([bsz, n_sent], dtype=torch.int32).cuda() * self.dictionary.indices['F']
        pad = torch.ones_like(out).int() * self.dictionary.indices['<pad>']
        false = torch.ones_like(out) * self.dictionary.indices['F']

        topk = pr.argsort(dim=-1)[:, -self.topk:]
        # topk = torch.stack([torch.tensor([0, 1, 2]).type_as(t) for t in topk])

        for idx in range(bsz):
            out[idx, topk[idx]] = self.dictionary.indices['T']
        # out = false
        out = torch.where(pr<=0, false, out)
        out = torch.where(mask==0, out, pad).int()
        # print(out)
        return out, weights
    '''
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


@register_model_architecture('extract_pacsum', 'extract_pacsum')
def base_architecture(args):
    # args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    # args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    # args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    # args.encoder_layers = getattr(args, 'encoder_layers', 6)
    # args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    # args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    # args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    # args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    # args.decoder_layers = getattr(args, 'decoder_layers', 6)
    # args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)

# # Medium size transformer
# @register_model_architecture('extract_sum_roberta_long_transformer', 'extract_sum_roberta_long_transformer_medium')
# def transformer_medium(args):
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
#     args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
#     args.dropout = getattr(args, 'dropout', 0.1)
#     base_architecture(args)


# # Medium size transformer
# @register_model_architecture('extract_sum_roberta_long_transformer', 'extract_sum_roberta_long_transformer_large')
# def transformer_medium(args):
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4086)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
#     args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
#     args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4086)
#     args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
#     args.dropout = getattr(args, 'dropout', 0.1)
#     base_architecture(args)
