# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('sents_recovery_after_shuffle')
class SentsRecoveryAfterShuffle(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.src_padding_idx = task.source_dictionary.pad()
        self.args = args

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--shuffle-weight', default=0.1, type=float)
        parser.add_argument('--mask-weight', default=0.9, type=float)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        # this is for the shuffle sentence loss
        lprobs_shuffle = model.get_normalized_probs(net_output, log_probs=True, idx=0)
        lprobs_shuffle = lprobs_shuffle.view(-1, lprobs_shuffle.size(-1))
        target_shuffle = model.get_targets(sample, net_output).view(-1, 1)
        # this is for padding mask
        non_pad_mask_shuffle = target_shuffle.ne(self.src_padding_idx)
        nll_loss_shuffle = -lprobs_shuffle.gather(dim=-1, index=target_shuffle)[non_pad_mask_shuffle]
        smooth_loss_shuffle = -lprobs_shuffle.sum(dim=-1, keepdim=True)[non_pad_mask_shuffle]

        # this is for the masked sentces loss
        lprobs_mask = model.get_normalized_probs(net_output, log_probs=True, idx=1)
        lprobs_mask = lprobs_mask.view(-1, lprobs_mask.size(-1))
        target_mask = model.get_targets(sample, net_output, 'mask_target').view(-1, 1)
        # this is for padding mask
        non_pad_mask_mask = target_mask.ne(self.src_padding_idx)
        nll_loss_mask = -lprobs_mask.gather(dim=-1, index=target_mask)[non_pad_mask_mask]
        smooth_loss_mask = -lprobs_mask.sum(dim=-1, keepdim=True)[non_pad_mask_mask]


        if reduce:
            nll_loss_shuffle = nll_loss_shuffle.sum()
            smooth_loss_shuffle = smooth_loss_shuffle.sum()
            nll_loss_mask = nll_loss_mask.sum()
            smooth_loss_mask = smooth_loss_mask.sum()
        eps_i_shuffle = self.eps / lprobs_shuffle.size(-1)
        eps_i_mask = self.eps / lprobs_mask.size(-1)
        loss_shuffle = (1. - self.eps) * nll_loss_shuffle + eps_i_shuffle * smooth_loss_shuffle
        loss_mask = (1. - self.eps) * nll_loss_mask + eps_i_mask * smooth_loss_mask
        loss = loss_shuffle * self.args.shuffle_weight  + loss_mask * self.args.mask_weight
        nll_loss = nll_loss_mask
        assert reduce

        # ntokens here are number of masked sentence tokens
        # sample_size = sample['target'].size(0) if self.args.sentence_avg else sample.get('sample_size', sample['ntokens'])
        sample_size = sample['shuffle_ntokens'] * self.args.shuffle_weight + sample['ntokens'] * self.args.mask_weight

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
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
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'sample_size': sample_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
        }
