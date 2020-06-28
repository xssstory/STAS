
import os
import torch
import numpy as np
from fairseq import options
from fairseq.data import (
    data_utils, GPT2Dictionary, PointerFlexibleDictionary, LanguagePairDataset, indexed_dataset,
    IndexedCachedDataset,
    IndexedRawTextDataset,
    SentsPermutationWithMaskDataset
)

from . import FairseqTask, register_task

@register_task('sents_permutation_with_mask')
class SentsPermutationWithMaskTask(FairseqTask):

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
        parser.add_argument('--left-pad-source', default='False', type=str, metavar='BOOL',
                            help='pad the source on the left (default: False)')
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
        parser.add_argument('--trigram-block', default='True', type=str, metavar='BOOL',
                            help='use trigram block during evaluation')
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

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.max_src_size = 0
        self.max_tgt_size = 0
        self.run_dummy_batch = True # set to False for test

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
        src_dict = GPT2Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.source_lang)))
        idx = src_dict.add_special_token('<sent_mask>')
        print('<sent_mask> id = {}, token = {}'.format(idx, src_dict[idx]))
        print('<mask> id is', src_dict.index('<mask>'))
        print('<sent_mask> id is', src_dict.index('<sent_mask>'))

        # tgt_dict = FlexibleDictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.target_lang)))
        # generate the tgt_dict 
        tgt_dict = PointerFlexibleDictionary(args.max_doc_length, specialTokens=[('EOS', '</s>'), ('PAD', '<pad>'), ('UNK', '<unk>'), ('BOS', '<s>')])

        assert tgt_dict.index('0') == 0
        print('| WARNING: idx should should match the context in the tgt dict')
        # if args.predict_arch == 'pointer_net':
        #     assert tgt_dict.eos() == args.max_doc_length

        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, shuffle=True, **kwargs):
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

        # tgt_dataset = indexed_dataset(prefix + tgt, self.tgt_dict)

        rng = np.random.RandomState(self.args.seed)

        # need to be updated with extractive summarization dataset
        self.datasets[split] = SentsPermutationWithMaskDataset(
            src_dataset, src_dataset.sizes, self.src_dict,
            None, None, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            shuffle=shuffle,
            is_poniter_net=(self.args.predict_arch == 'pointer_net'),
            max_sent_len=self.args.max_sent_length,
            max_doc_len=self.args.max_doc_length,
            masked_sent_prob=self.args.masked_sent_prob,
            max_predictions_per_doc=self.args.max_predictions_per_doc,
            rng=rng,
            doc_sizes=None,
        )

    def load_pretrained_model(self, model, state_file_name):
        from torch.serialization import default_restore_location
        state = torch.load(state_file_name, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        params = state['model']

        # non_encoder_param_names = [k for k in params.keys() if not k.startswith('encoder')]
        # for nk in non_encoder_param_names:
        #     del params[nk]

        non_decoder_param_names = [k for k in params.keys() if not k.startswith('decoder')]
        for nk in non_decoder_param_names:
            del params[nk]

        enc_cnt = 0
        non_enc_cnt = 0
        for k in params.keys():
            if not k.startswith('encoder'):
                print(k)
                non_enc_cnt += 1
            else:
                enc_cnt += 1
        print('enc_cnt = %d, non_enc_cnt = %d'%(enc_cnt, non_enc_cnt))
        model.load_state_dict(params, strict=False)
        print('*** *** load pretrained doc encoder from {} done! *** ***'.format(state_file_name))

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
