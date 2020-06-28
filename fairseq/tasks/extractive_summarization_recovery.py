
import os
import numpy as np
import torch
from fairseq import options
from fairseq.data import (
    data_utils, GPT2Dictionary, FlexibleDictionary, LanguagePairDataset, indexed_dataset,
    ExtractSumRecoveryDataset,
    IndexedRawTextDataset,
    IndexedCachedDataset,
)
from fairseq.data.bert_dictionary import BertDictionary
from . import FairseqTask, register_task


@register_task('extractive_summarization_recovery')
class ExtractiveSummarizationRecovery(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=40960, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=40960, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--ncpu-eval', default=2, type=int, metavar='N',
                            help='number of CPUs during rouge evaluation')
        parser.add_argument('--topk-sent-eval', default=3, type=int, metavar='N',
                            help='number of sentences selected during rouge evaluation')
        parser.add_argument('--raw-valid', default=None, metavar='RAW_VALID',
                            help='raw valid set')
        parser.add_argument('--raw-test', default=None, metavar='RAW_TEST',
                            help='raw test set')
        parser.add_argument('--max-sent-length', default=50, type=int, metavar='N',
                            help='max number of tokens a source document sentence can have')
        parser.add_argument('--max-doc-length', default=30, type=int, metavar='N',
                            help='max number of sentences a source document can have')

        parser.add_argument('--masked-sent-prob', default=0.15, type=float, help='prob to predict masked lm')
        parser.add_argument('--max-predictions-per-doc', default=5, type=int, help='maximum number of masked sentences per doc')

        parser.add_argument('--init-from-pretrained-doc-model', default='False', type=str, metavar='BOOL',
                            help='init model from a pretrained model')
        parser.add_argument('--pretrained-doc-model-path', default=None, metavar='PRETRAINED_PATH',
                            help='pretrained doc model path')
        parser.add_argument('--trigram-block', default='True', type=str)
        parser.add_argument('--mask-other-sents', default='False', type=str)
        parser.add_argument('--max-roberta-position', default=512, type=int)

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.max_src_size = 0
        self.max_tgt_size = 0
        self.run_dummy_batch = True
        # self.run_dummy_batch = False # just for test

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        args.trigram_block = options.eval_bool(args.trigram_block)
        args.init_from_pretrained_doc_model = options.eval_bool(args.init_from_pretrained_doc_model)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data)
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        if args.roberta_model.startswith('roberta'):
            src_dict = GPT2Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.source_lang)))
        else:
            src_dict = BertDictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.source_lang)))
        idx = src_dict.add_special_token('<sent_mask>')
        print('<sent_mask> id = {}, token = {}'.format(idx, src_dict[idx]))
        print('<mask> id is', src_dict.index('<mask>'))
        print('<sent_mask> id is', src_dict.index('<sent_mask>'))

        tgt_dict = FlexibleDictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.target_lang)))
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def create_doc_size_file(self, doc_dataset, sent_sep_idx, doc_size_file):
        with open(doc_size_file, 'w', encoding='utf8') as fout:
            print('dataset size', len(doc_dataset))
            for i in range(len(doc_dataset)):
                src_doc = doc_dataset[i]
                # src_doc = self.src[index]

                istart = 0
                max_sent_len = 0
                doc_nsent = 0
                for i in range(len(src_doc)):
                    if src_doc[i] == sent_sep_idx or i == len(src_doc) - 1:
                        sent_len = i - istart
                        if src_doc[i] != sent_sep_idx:
                            sent_len += 1
                        max_sent_len = max(max_sent_len, sent_len)
                        istart = i+1
                        doc_nsent += 1

                fout.write('{}\t{}\n'.format(doc_nsent, max_sent_len))
                fout.flush()

    def load_doc_size_file(self, doc_size_file):
        doc_sizes = []
        for line in open(doc_size_file, encoding='utf8'):
            fds = line.strip().split()
            assert len(fds) == 2, 'size file MUST have two fileds'
            doc_sizes.append( (int(fds[0]), int(fds[1])) )
        print('load doc size done', len(doc_sizes))
        return doc_sizes

    def load_dataset(self, split, shuffle=True):
        """Load a dataset split."""

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            return indexed_dataset.dataset_exists(filename, impl=self.args.dataset_impl)

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang
        if split_exists(split, src, tgt, src, self.args.data):
            prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
        elif split_exists(split, tgt, src, src, self.args.data):
            prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
        else:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, self.src_dict, self.args.dataset_impl)
        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, self.tgt_dict, self.args.dataset_impl)

        rng = np.random.RandomState(self.args.seed)

        # get doc size information
        assert isinstance(src_dataset, IndexedCachedDataset), 'currently only support IndexedInMemoryDataset'
        src_path = prefix + src

        # need to be updated with extractive summarization dataset
        self.datasets[split] = ExtractSumRecoveryDataset(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes if tgt_dataset is not None else None, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            shuffle=shuffle,
            max_sent_len=self.args.max_sent_length,
            max_doc_len=self.args.max_doc_length,
            masked_sent_prob=self.args.masked_sent_prob,
            max_predictions_per_doc=self.args.max_predictions_per_doc,
            rng=rng,
            doc_sizes=None,
            mask_other_sents=eval(self.args.mask_other_sents),
            max_tokens_len=self.args.max_roberta_position,
            bert_model= self.args.roberta_model,
        )


    def load_pretrained_model(self, model, state_file_name):
        from torch.serialization import default_restore_location
        state = torch.load(state_file_name, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        if state['args'].fp16 != model.args.fp16:
            print('change the fp16 in model to {}'.format(state['args'].fp16))
            model.args.fp16 = state['args'].fp16
        params = state['model']

        print('num params', len(list(params.keys())))
        tobe_del_param_names = [k for k in params.keys() if k.startswith('decoder.out_proj')]
        for nk in tobe_del_param_names:
            del params[nk]

        print('num params after removing', len(list(params.keys())) )

        model.load_state_dict(params, strict=False)
        print('*** *** load pretrained doc encoder done! *** ***')


    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict

    def clear_cuda(self, sample):
        src_size = sample['net_input']['src_tokens'].numel()
        tgt_size = sample['target'].numel()
        if src_size > self.max_src_size or tgt_size > self.max_tgt_size:
            torch.cuda.empty_cache()
            if src_size > self.max_src_size:
                self.max_src_size = src_size
            if tgt_size > self.max_tgt_size:
                self.max_tgt_size = tgt_size
