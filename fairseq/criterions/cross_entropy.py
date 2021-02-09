# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('cross_entropy')
class CrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.task = task

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # import pdb
        # pdb.set_trace()
        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
                          reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],

            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        # 测试查看：运行时刻模型输出分布最大概率采样产生的目标语言序列
        # tgt = sample['target']
        # hypo = torch.max(lprobs,1)[1].reshape(*tgt.shape)
        # print(tgt, hypo, sep="\n")

        # tgt_str = self.task.tgt_dict.string(tgt, True, escape_unk=True)
        # hypo_str = self.task.tgt_dict.string(hypo, True, escape_unk=True)

        # pre_str = self.task.tgt_dict.string(sample['net_input']['prev_output_tokens'], True, escape_unk=True)
        # for t,h,p in zip(tgt_str.split("\n"),hypo_str.split("\n"),pre_str.split("\n")):
        #     print(f"T: {t}")
        #     print(f"H: {h}")
        #     print(f"P: {p}")
        #     print("-"*20)

        # from fairseq.sequence_generator import SequenceGenerator
        
        # translator = SequenceGenerator(
        #     [model], self.task.target_dictionary, beam_size=5)
        
        # print("-"*20, "by sequence generator", "-"*20)
        # print(translator.generate_by_a_sample(sample))
        
        # import pdb
        # pdb.set_trace()


        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
