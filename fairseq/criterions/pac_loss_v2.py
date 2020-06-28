# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch

from fairseq import utils
from torch.nn import functional as F

from . import FairseqCriterion, register_criterion


@register_criterion('pac_loss_v2')
class PacLossV2(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])

        selected_sent_embedding = net_output['select_sent_embedding']
        positive_sent_context = net_output['positive_sent_context']
        # negative_sent_embedding = net_output['negative_sent_embedding']
        negative_sent_context = net_output['negative_sent_context']

        bsz = selected_sent_embedding.shape[0]

        loss = 0
        for idx in range(bsz):
            loss -= F.logsigmoid(selected_sent_embedding[idx].matmul(positive_sent_context[idx].transpose(-1, -2))).sum()
            # index = t.LongTensor([[i for i in range(negative_sent_embedding.size(-2))]])
            # loss -= F.logsigmoid(negative_sent_embedding[idx].matmul(negative_sent_context[idx].transpose(-1, -2))).gather(0, index).mean()
            loss -= F.logsigmoid(-selected_sent_embedding[idx].matmul(negative_sent_context[idx].transpose(-1, -2))).mean()

        sample_size = len(sample['id'])
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(loss.data) if reduce else loss.data,
            # 'ns_nll_loss': utils.item(ns_nll_loss.data) if reduce else ns_nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'sample_size': sample_size,
            "nsentences": nsentences,
            "ntokens": ntokens,
        }
