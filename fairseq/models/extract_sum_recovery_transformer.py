
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# from sklearn.cluster import KMeans
import numpy as np

from fairseq import utils

from fairseq.modules import (
    LearnedPositionalEmbedding, MultiheadAttention,
    SinusoidalPositionalEmbedding
)

from fairseq.modules.attn_score_bert_encoder import AttnScoreBertEncoder
from .transformer_sents_decoder import TransformerSentDecoder
from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqModel,
    register_model, register_model_architecture,
)
from fairseq import utils
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


@register_model('extract_sum_recovery')
class ExtractSumRecoveryTransformer(FairseqModel):
    def __init__(self, encoder, decoder, src_dict, tgt_dict, args):
        super().__init__(encoder, decoder)
        self.topk = args.topk
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
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
        parser.add_argument('--roberta-model', default='roberta-base', choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-base-chinese'], help="RoBERTa pre-trained model selected in the list: roberta-base, "
                                "roberta-large")
        parser.add_argument('--sentence-transformer-arch', default='fairseq', help="sentence level transformer architecture [fairseq, bert]")
        parser.add_argument('--bert-no-decay', default=False, action='store_true', help="no decay for bias and LayerNorm.weight")
        parser.add_argument('--attn-type',  choices=['attn_score', 'attn_prob', 'choose_head'], default='attn_prob')
        # parser.add_argument('--recovery-score-type', default='negative', choices=['negative', 'positive'])
        parser.add_argument('--lam', default='(0.5,0.5,0)', type=str)
        parser.add_argument('--topk', default=3, type=int)
        parser.add_argument('--need-cluster', default=False, action='store_true')
        parser.add_argument('--ignore-sent-mask', default=False, action='store_true')
        parser.add_argument('--sents-per-cluster', default=5, type=int)
        parser.add_argument('--recovery-thresh', default=-1, type=float)

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
            # note the decoder has the same vocabulary size
            decoder_embed_tokens = build_embedding(
                src_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, src_dict, tgt_dict, decoder_embed_tokens)
        return cls(encoder, decoder, src_dict, tgt_dict, args)

    def single_forward(self, src_tokens, segment_ids, doc_pad_mask, doc_pos_tok, cls_pos, masked_sent_positions, prev_output_tokens, token_mask):
        encoder_out = self.encoder(src_tokens, segment_ids, doc_pad_mask, doc_pos_tok, cls_pos, token_mask)
        # if self.args.attn_type == 'choose_head':
        #     decoder_out = [self.decoder(encoder_out, mamasked_sent_positions, prev_output_tokens, head) for head]
        # else:
        decoder_out = self.decoder(encoder_out, masked_sent_positions, prev_output_tokens)
        return encoder_out, decoder_out
    
    def cluster(self, k, embedding):
        device = embedding.device
        nsents, _ = embedding.shape
        embedding = embedding.cpu().numpy()
        kmeans = KMeans(n_clusters=k).fit(embedding)
        label = kmeans.labels_
        centers = kmeans.cluster_centers_
        center_per_sent = centers[label]
        distances = ((embedding - center_per_sent) ** 2).sum(axis=-1)
        all_distances = ((embedding[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=-1)
        cluster_score = (1 + distances) ** -1 / ((1 + all_distances) ** -1).sum(axis=-1) 

        return torch.from_numpy(label).to(device), torch.from_numpy(distances).to(device), torch.from_numpy(cluster_score).to(device)
    
    def get_cluster_topk(self, nsents, cluster_labels, cluster_distances, doc_pad_mask, topk, bsz):
        real_topk_list = []
        for idx, (cluster_label, cluster_distance) in enumerate(zip(cluster_labels, cluster_distances)):
            # out[idx, topk[idx]] = self.tgt_dict.index('T')
            n_sent = nsents[idx]
            n_cluster = math.ceil(n_sent / self.args.sents_per_cluster)

            # cluster_label, cluster_distance = self.cluster(n_cluster, encoder_out_without_mask['encoder_out'][idx][doc_pad_mask[idx]==0])
            tmp_out = torch.ones(max(nsents), device=topk.device) * self.tgt_dict.index('F')
            # tmp_out = out.new(out.shape[-1]).copy_(out[idx])
            tmp_out[topk[idx]] = self.tgt_dict.index('T')
            tmp_out = tmp_out[doc_pad_mask[idx]==0]

            # cluster_label = cluster_label.view(1, -1)
            # cluster_distance = cluster_distance.view(1, -1)
            select_cluster = None
            max_overlap = 0
            for cluster in range(n_cluster):
                overlap = ((tmp_out == self.tgt_dict.index('T')) & (cluster_label == cluster)).sum()
                if overlap > max_overlap:
                    max_overlap = overlap
                    select_cluster = cluster
                    if max_overlap.item() >= self.topk / 2:
                        break
            if max_overlap < 3:
                # don't use cluster
                real_topk = topk[idx]
            else:
                cluster_distance.masked_fill_(cluster_label!=select_cluster, math.inf)
                real_topk = cluster_distance.argsort(dim=-1)[:self.topk]
            real_topk_list.append(real_topk)
        return real_topk_list
    
    def forward(self, bsz, max_nsents, masked_docs_generator, doc_without_mask):

        def get_attn_score(attn_weights, idx, ignore_mask=True):
            attn_weights = F.relu(attn_weights)
            attn_score = attn_weights[:, idx, :].squeeze(dim=1)
            if ignore_mask:
                attn_score[:, idx] = 0
            return attn_score
        
        def get_attn_score_itera(attn_layer_list, idx, layer_id, nsents):
            attn_layer = F.relu(attn_layer_list[layer_id])
            attn_score = attn_layer[:, idx, :].squeeze(dim=1)
            attn_next_weights = attn_score[:, idx].clone().view(-1, 1) > 1 / torch.Tensor(nsents).view(-1, 1).to(attn_score.device)
            attn_next_weights = attn_next_weights.float()
            attn_score[:, idx] = 0

            if layer_id == 0:
                return attn_score
            else:
                return attn_score + attn_next_weights * get_attn_score_itera(attn_layer_list, idx, layer_id-1, nsents)
        
        def get_attn_score_for_layers(attn_layer_list, idx, offset=0.5):
            attn_score = attn_layer_list[-1][:, idx, :].squeeze(dim=1)
            attn_next_weights = attn_score[:, idx].clone().view(-1, 1) + offset
            attn_next_weights.masked_fill_(attn_next_weights > 1, 1)
            attn_score[:, idx] = 0
            for subidx in range(2, len(attn_layer_list) + 1):
                tmp_weights = attn_next_weights
                attn_next_score = attn_layer_list[-subidx][:, idx, :].squeeze(dim=1)
                attn_next_weights = attn_next_score[:, idx].clone().view(-1, 1) + offset
                attn_next_weights.masked_fill_(attn_next_weights > 1, 1)
                attn_next_score[:, idx] = 0
                attn_score += attn_next_score * tmp_weights
            return attn_score

        lam = eval(self.args.lam)
        if len(lam) == 2:
            lam1, lam2 = lam
            lam3 = 0
        else:
            lam1, lam2, lam3 = lam
        # scores = [torch.zeros([bsz, max_nsents], dtype=torch.float32).cuda() for _ in range(self.args.encoder_attention_heads + 4)]
        scores = [torch.zeros([bsz, max_nsents], dtype=torch.float32).cuda() for _ in range(3) ]
        if self.args.fp16:
            scores = [score.half() for score in scores]
        outs = [torch.ones([bsz, max_nsents], dtype=torch.int32).cuda() * self.tgt_dict.index('F') for _ in scores]
        nsents = torch.tensor(doc_without_mask['nsents']).cuda().view(bsz, -1)
        doc_pad_mask = None
        token_mask = None
        # just for evaluation
        with torch.no_grad():

            for idx, masked_docs_input in enumerate(masked_docs_generator):
                masked_docs_input = utils.move_to_cuda(masked_docs_input)

                if doc_pad_mask is not None:
                    assert doc_pad_mask.equal(masked_docs_input['net_input']["doc_pad_mask"])
                    assert token_mask.equal(masked_docs_input['net_input']['token_mask'])
                else:
                    doc_pad_mask = masked_docs_input['net_input']["doc_pad_mask"].clone()
                    token_mask = masked_docs_input['net_input']['token_mask'].clone()
                if self.args.ignore_sent_mask:
                    masked_docs_input['net_input']["doc_pad_mask"][:, idx] = True

                encoder_out, decoder_out = self.single_forward(**masked_docs_input["net_input"])

                # if doc_pad_mask is not None:
                #     assert (doc_pad_mask ==  encoder_out["encoder_doc_mask"]).all()
                # else:
                #     doc_pad_mask = encoder_out["encoder_doc_mask"]

                lprobs = self.get_normalized_probs(decoder_out, log_probs=False, idx=0)        
                lprobs = lprobs.view(-1, lprobs.size(-1))
                if self.args.fp16:
                    lprobs = lprobs.half()

                # compute the revovery score for current sentence
                target = self.get_targets(masked_docs_input, decoder_out)
                # this is for padding mask
                non_pad_mask = target.ne(self.src_dict.pad_index)
                target = target.view(-1, 1)
                # if self.args.recovery_score_type == 'negative':
                #     recovery_score = -lprobs.gather(dim=-1, index=target).view(non_pad_mask.shape).masked_fill_(non_pad_mask==0, 0).sum(dim=-1) / non_pad_mask.sum(dim=-1).type_as(lprobs)
                # else:
                #     recovery_score = lprobs.gather(dim=-1, index=target).view(non_pad_mask.shape).masked_fill_(non_pad_mask==0, 0).sum(dim=-1) / non_pad_mask.sum(dim=-1).type_as(lprobs)
                recovery_score = lprobs.gather(dim=-1, index=target).view(non_pad_mask.shape).masked_fill_(non_pad_mask==0, 0).sum(dim=-1) / non_pad_mask.sum(dim=-1).type_as(lprobs)    
                recovery_score.masked_fill_(nsents<=idx, 0)

                # compute the attn score for other sentences
                attn_weights = encoder_out['attn_weights']
                attn_head_avg_head = [weight.mean(dim=1) for weight in attn_weights]
                attn_weights_all_layer = sum(attn_head_avg_head)
                attn_weights_avg_all_layer = attn_weights_all_layer / self.args.encoder_layers
                attn_weights_final_layer_avg = attn_head_avg_head[-1]
                attn_weights_final_layer_avg_scale = attn_weights_final_layer_avg * self.args.encoder_layers

                attn_final_layer = attn_weights[-1]
                attn_final_layer_heads = torch.split(attn_final_layer, 1, dim=1)
                attn_final_layer_heads = [head.squeeze(dim=1) for head in attn_final_layer_heads]
                attn_final_layer_heads = [head  * self.args.encoder_layers for head in attn_final_layer_heads]
                
                # attn_weights_list = attn_final_layer_heads + [attn_weights_all_layer, attn_weights_avg_all_layer, attn_weights_final_layer_avg_scale, attn_weights_final_layer_avg]
                # attn_weights_list = [attn_weights_all_layer, attn_weights_avg_all_layer]
                # attn_weights_list = [attn_weights_avg_all_layer, attn_weights_avg_all_layer * 2, attn_weights_avg_all_layer * 3, attn_weights_avg_all_layer * 4, attn_weights_avg_all_layer *6]
                attn_weights_list = [attn_weights_avg_all_layer * 3, attn_weights_avg_all_layer * 4]
                assert len(scores) == len(attn_weights_list) + 1
                # get attn_score in the attn list
                for j, attn_weights in enumerate(attn_weights_list):
                    attn_score = get_attn_score(attn_weights, idx)
                    scores[j][:, idx] += recovery_score.view(-1) * lam1
                    if self.args.recovery_thresh < 0:
                        scores[j] += attn_score * recovery_score * lam2
                    else:
                        scores[j] += attn_score * torch.where(recovery_score<self.args.recovery_thresh, torch.zeros_like(recovery_score), torch.ones_like(recovery_score)) * lam2
                    if self.args.need_cluster:
                        scores[j] += cluster_scores * lam3
                
                # get attn_score in the layer by layer
                scores[-1][:, idx] += recovery_score.view(-1) * lam1
                attn_score = get_attn_score_for_layers(attn_head_avg_head, idx)
                if self.args.recovery_thresh < 0:
                    scores[-1] += attn_score * recovery_score * lam2
                else:
                    scores[-1] += attn_score * torch.where(recovery_score<self.args.recovery_thresh, torch.zeros_like(recovery_score), torch.ones_like(recovery_score)) * lam2
                if self.args.need_cluster:
                    scores[-1] += cluster_scores * lam3

            for score, out in zip(scores, outs):
                score.masked_fill_(doc_pad_mask, score.min())
                out.masked_fill_(doc_pad_mask, self.tgt_dict.pad_index)
            topks = [score.argsort(dim=-1)[:, -self.topk:] for score in scores]
                
            for topk, out in zip(topks, outs):
                for idx in range(bsz):
                    out[idx, topk[idx]] = self.tgt_dict.index('T')

        return (outs, scores) if not self.args.need_cluster else (outs, scores, cluster_labels)


    def get_normalized_probs(self, net_output, log_probs, idx=0):
        """Get normalized probabilities (or log probs) from a net's output."""
        assert idx == 0 or idx == 1
        logits = net_output[idx].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def get_targets(self, sample, net_output, key='target'):
        """Get targets from either the sample or the net's output."""
        return sample[key]


class TransformerEncoder(FairseqEncoder):
    """Transformer encoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout

        # from pytorch_transformers import RobertaModel
        from fairseq.modules.roberta_causal_mask import RobertaCasulMaskModel, BertCasulMaskModel
        from pytorch_transformers.file_utils import PYTORCH_TRANSFORMERS_CACHE
        from pytorch_transformers import RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer

        if args.roberta_model.startswith('roberta'):
            self.roberta = RobertaCasulMaskModel.from_pretrained(args.roberta_model,
                    cache_dir=PYTORCH_TRANSFORMERS_CACHE / 'distributed_{}'.format(args.distributed_rank))
            self.config = RobertaConfig.from_pretrained(args.roberta_model)
            self.tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model)
        else:
            self.roberta = BertCasulMaskModel.from_pretrained(args.roberta_model,
                    cache_dir=PYTORCH_TRANSFORMERS_CACHE / 'distributed_{}'.format(args.distributed_rank))
            self.config = BertConfig.from_pretrained(args.roberta_model)
            self.tokenizer = BertTokenizer.from_pretrained(args.roberta_model)
        self.config.output_attentions = True
        self.roberta.pooler.dense.weight.requires_grad = False
        self.roberta.pooler.dense.bias.requires_grad = False

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx

        # self.embed_tokens = embed_tokens
        # self.embed_scale = math.sqrt(embed_dim)

        self.args = args

        # if args.sentence_transformer_arch == 'fairseq':
        #     self.padding_idx = embed_tokens.padding_idx

        #     self.sent_embed_positions = PositionalEmbedding(
        #         1024, embed_dim, self.padding_idx,
        #         left_pad=False,
        #         learned=args.encoder_learned_pos,
        #     )

        #     self.doc_layers = nn.ModuleList([])
        #     self.doc_layers.extend([
        #         TransformerEncoderLayer(args)
        #         for i in range(args.encoder_layers)
        #     ])
        if args.sentence_transformer_arch == 'bert':
            # from pytorch_transformers import RobertaConfig, RobertaTokenizer

            # self.config = RobertaConfig.from_pretrained(args.roberta_model)
            # self.config.output_attentions = True
            # self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

            if args.max_roberta_position > 512:
                self.roberta.expand_position_embedding(args.max_roberta_position, self.config.initializer_range)

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

    def forward(self, src_tokens, segment_ids, doc_pad_mask, doc_pos_tok, cls_pos, attention_mask=None):
        # if self.args.sentence_transformer_arch == 'fairseq':
        #     bsz, seqlen = src_tokens.size()

        #     # compute padding mask
        #     attention_mask = src_tokens.ne(self.padding_idx)
        #     # enc_hids, _ = self.bert(src_tokens, segment_ids, attention_mask, output_all_encoded_layers=False)
        #     all_hids = self.roberta(src_tokens, segment_ids, attention_mask)
        #     # print('all_hids', all_hids.size())
        #     enc_hids = all_hids[0]
        #     doc_pos = self.sent_embed_positions(doc_pos_tok)

        #     sent_repr = get_sent_end_repr(enc_hids, cls_pos)

        #     sent_repr = sent_repr + doc_pos
        #     # n_sent x bsz x C
        #     sent_repr = sent_repr.transpose(0, 1)
        #     for doc_layer in self.doc_layers:
        #         sent_repr = doc_layer(sent_repr, doc_pad_mask)

        #     return {
        #         'encoder_out': sent_repr,  # n_sent x bsz x C
        #         'encoder_padding_mask': doc_pad_mask,  # bsz x n_sent
        #     }
        if self.args.sentence_transformer_arch == 'bert':
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


class TransformerDecoder(TransformerSentDecoder):
    """Transformer decoder."""

    def __init__(self, args, src_dictionary, dictionary, embed_tokens, left_pad=False):
        '''
        The decoder has two parts:
            1) a transformer decoder to predict masked sentences
            2) a decoder to predict sentence labels
        '''
        super().__init__(args, src_dictionary, dictionary, embed_tokens, left_pad)

        # this part is for sentence label prediction
        self.out_proj = Linear(embed_tokens.embedding_dim, len(dictionary))

    def forward_sent_label(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out_proj(x)

    def forward(self, encoder_out, masked_sent_positions, prev_output_tokens, incremental_state=None):

        # n_sent x bsz x C
        x = encoder_out['encoder_out']
        # n_sent x bsz x C -> bsz x n_sent x C
        if self.sentence_transformer_arch == 'fairseq':
            x = x.transpose(0, 1)

        # predict sentence label
        sent_label = self.forward_sent_label(x)

        # predict masked sentence
        masked_sent = self.forward_masked_sent(x, masked_sent_positions, prev_output_tokens, incremental_state)

        return masked_sent, sent_label


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


@register_model_architecture('extract_sum_recovery', 'extract_sum_recovery')
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
@register_model_architecture('extract_sum_recovery', 'extract_sum_recovery_medium')
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    args.dropout = getattr(args, 'dropout', 0.1)
    base_architecture(args)

@register_model_architecture('extract_sum_recovery', 'extract_sum_recovery_base')
def transformer_vaswani_wmt_en_de_big(args):
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
@register_model_architecture('extract_sum_recovery', 'extract_sum_recovery_large')
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
