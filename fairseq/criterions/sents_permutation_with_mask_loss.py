# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('sents_perm_with_mask')
class SentsPermWithMaskLoss(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.tgt_padding_idx = task.target_dictionary.pad()
        self.src_padding_idx = task.source_dictionary.pad()
        self.masked_sent_loss_weight = args.masked_sent_loss_weight
        self.sent_perm_weight = args.sent_perm_weight

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--masked-sent-loss-weight', default=0.5, type=float, metavar='D',
                            help='weight for masked sentence predition')
        parser.add_argument('--sent-perm-weight', default=0.5, type=float, metavar='D',
                            help='weight for masked sentence predition')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])

        #perm loss
        lprobs_perm = model.get_normalized_probs(net_output, log_probs=True, index=0)
        lprobs_perm = lprobs_perm.view(-1, lprobs_perm.size(-1))
        target_perm = model.get_targets(sample, net_output, 'target_perm')
        # this is for padding mask
        non_pad_mask_perm = target_perm.ne(self.tgt_padding_idx).view(-1, 1)
        target_perm.masked_fill_(target_perm==self.tgt_padding_idx, 0)
        nll_loss_perm = -lprobs_perm.gather(dim=-1, index=target_perm.view(-1, 1))[non_pad_mask_perm]
        smooth_loss_perm = -lprobs_perm.sum(dim=-1, keepdim=True)[non_pad_mask_perm]

        # mask loss
        lprobs = model.get_normalized_probs(net_output, log_probs=True, index=1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        # this is for padding mask
        non_pad_mask = target.ne(self.src_padding_idx)
        nll_loss_mask = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss_mask = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]


        if reduce:
            nll_loss_mask = nll_loss_mask.sum()
            smooth_loss_mask = smooth_loss_mask.sum()
            nll_loss_perm = nll_loss_perm.sum()
            smooth_loss_perm = smooth_loss_perm.sum()
        nll_loss = nll_loss_mask * self.masked_sent_loss_weight + nll_loss_perm * self.sent_perm_weight
        smooth_loss = smooth_loss_mask * self.masked_sent_loss_weight + smooth_loss_perm * self.sent_perm_weight

        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss_mask.data) if reduce else nll_loss_mask.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample.get('nsentences', None) or sample['net_input']['src_tokens'].size(0) if sample is not None else 0,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)

        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) ,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'sample_size': sample_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
        }
