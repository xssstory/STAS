#!/usr/bin/env python3
#

import argparse
from itertools import zip_longest
import os, torch
import shutil

from fairseq.data import indexed_dataset, dictionary, flexible_dictionary, gpt2_dictionary
from fairseq.tokenizer import Tokenizer, tokenize_line


def get_parser():
    parser = argparse.ArgumentParser(
        description='Data pre-processing: Create dictionary and store data in binary format')
    parser.add_argument('-s', '--source-lang', default=None, metavar='SRC', help='source language')
    parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET', help='target language')
    parser.add_argument('--trainpref', metavar='FP', default=None, help='target language')
    parser.add_argument('--validpref', metavar='FP', default=None, help='comma separated, valid language prefixes')
    parser.add_argument('--testpref', metavar='FP', default=None, help='comma separated, test language prefixes')
    parser.add_argument('--destdir', metavar='DIR', default='data-bin', help='destination dir')
    parser.add_argument('--thresholdtgt', metavar='N', default=0, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--thresholdsrc', metavar='N', default=0, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--tgtdict', metavar='FP', help='reuse given target dictionary')
    parser.add_argument('--srcdict', metavar='FP', help='reuse given source dictionary')
    parser.add_argument('--nwordstgt', metavar='N', default=-1, type=int, help='number of target words to retain')
    parser.add_argument('--nwordssrc', metavar='N', default=-1, type=int, help='number of source words to retain')
    parser.add_argument('--output-format', metavar='FORMAT', default='binary', choices=['binary', 'raw'],
                        help='output format (optional)')
    parser.add_argument('--joined-dictionary', action='store_true', help='Generate joined dictionary')
    parser.add_argument('--only-source', action='store_true', help='Only process the source language')
    parser.add_argument('--padding-factor', metavar='N', default=8, type=int,
                        help='Pad dictionary size to be multiple of N')
    parser.add_argument('--max-num-sentences', metavar='N', default=30, type=int, help='maximum number of sentences in an article')
    parser.add_argument('--max-num-words', metavar='N', default=100, type=int, help='maximum number of sentences in an article')

    return parser


def main(args):
    from fairseq import utils
    utils.xpprint(args)
    os.makedirs(args.destdir, exist_ok=True)
    target = not args.only_source

    def build_dictionary(filenames):
        d = dictionary.Dictionary()
        for filename in filenames:
            Tokenizer.add_file_to_dictionary(filename, d, tokenize_line)
        return d

    def build_dictionary_label(filenames):
        d = flexible_dictionary.FlexibleDictionary([('PAD', '<pad>')])
        for filename in filenames:
            Tokenizer.add_file_to_dictionary(filename, d, tokenize_line, append_eos=False)
        return d

    def train_path(lang):
        return '{}{}'.format(args.trainpref, ('.' + lang) if lang else '')

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += f'.{lang}'
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path('dict', lang) + '.txt'

    def dataset_dest_path(output_prefix, lang, extension):
        base = f'{args.destdir}/{output_prefix}'
        lang_part = f'.{args.source_lang}-{args.target_lang}.{lang}' if lang is not None else ''
        return f'{base}{lang_part}.{extension}'

    assert args.srcdict is not None, 'where is the Bert Dict!'
    if args.srcdict:
        src_dict = gpt2_dictionary.GPT2Dictionary.load(args.srcdict)
        src_dict.save(dict_path(args.source_lang))
        print('load bert dict from {} | size {}'.format(args.srcdict, len(src_dict)))
    else:
        assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
        src_dict = build_dictionary([train_path(args.source_lang)])
    if target:
        if args.tgtdict:
            tgt_dict = flexible_dictionary.FlexibleDictionary.load(args.tgtdict)
            print('load label dict from {} | size {}'.format(args.tgtdict, len(tgt_dict)))
        else:
            assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
            tgt_dict = build_dictionary_label([train_path(args.target_lang)])
            print('build target dict from {} done'.format(train_path(args.target_lang)))

    src_dict.save(dict_path(args.source_lang))
    if target:
        if not args.joined_dictionary:
            tgt_dict.finalize(
                threshold=args.thresholdtgt,
                nwords=args.nwordstgt,
                padding_factor=1,
            )
        tgt_dict.save(dict_path(args.target_lang))

    def make_binary_dataset(input_prefix, output_prefix, lang, append_eos=False):
        if lang == args.target_lang:
            dict = flexible_dictionary.FlexibleDictionary.load(dict_path(lang))
        else:
            # dict = bert_dictionary.BertDictionary.load(dict_path(lang))
            dict = gpt2_dictionary.GPT2Dictionary.load(dict_path(lang))

        print('| [{}] Dictionary: {} types | {} types (for real)'.format(lang, len(dict) - 1, len(dict)))

        ds = indexed_dataset.IndexedDatasetBuilder(dataset_dest_path(output_prefix, lang, 'bin'))

        def consumer(tensor):
            ds.add_item(tensor)

        input_file = '{}{}'.format(input_prefix, ('.' + lang) if lang is not None else '')
        if lang == args.target_lang:
            res = Tokenizer.binarize(input_file, dict, consumer, append_eos=append_eos)
            print('| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}'.format(
                lang, input_file, res['nseq'], res['ntok'],
                100 * res['nunk'] / res['ntok'], dict.unk_word if hasattr(dict, 'unk_word') else '<no_unk_word>'))
        else:
            # read article
            # from pytorch_pretrained_bert.tokenization import BertTokenizer
            # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            from pytorch_transformers import RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

            def penn_token2orig_token(sent):
                # -LRB- -RRB- -LSB- -RSB- -LCB- -RCB-
                '''
                penn2orig = {"``":'"', "''": '"',
                             "-LRB-": '(', "-RRB-": ')',
                             "-LSB-":'[', "-RSB-":']',
                             "-LCB-":'{', "-RCB-":'}'}
                '''
                penn2orig = {"-LRB-": '(', "-RRB-": ')',
                             "-LSB-": '[', "-RSB-": ']',
                             "-LCB-": '{', "-RCB-": '}',
                             "-lrb-": '(', "-rrb-": ')',
                             "-lsb-": '[', "-rsb-": ']',
                             "-lcb-": '{', "-rcb-": '}',}
                words = sent.strip().split()
                words = [wd if not wd in penn2orig else penn2orig[wd] for wd in words]
                return ' '.join(words)

            num_token, num_unk_token = 0, 0
            num_seq = 0
            skip_line = 0
            for line in open(input_file, encoding='utf8'):
                sents = line.strip().split('<S_SEP>')
                sents = sents[0:args.max_num_sentences]
                sents = [' '.join(sent.strip().split()[0:args.max_num_words]) for sent in sents]
                # print(sents)
                sents = [tokenizer.tokenize(penn_token2orig_token(sent)) for sent in sents]
                article_wids = []
                for i, sent in enumerate(sents):
                    # sometimes there are too many tokens
                    MAXLEN = 500
                    if len(sent) > MAXLEN:
                        # sent = sent[0:MAXLEN]
                        print(' '.join(sent))
                        skip_line += 1
                        print(skip_line)
                        continue
                    if i != 0:
                        article_wids.append( dict.sep_index )
                    wids = tokenizer.convert_tokens_to_ids(sent)
                    # wids_vocab = [dict.index(word) for word in sent]
                    # assert wids == wids_vocab, 'word indices should be the same!'
                    article_wids.extend(wids)
                    for wid in wids:
                        if wid == dict.unk_index:
                            num_unk_token += 1
                        num_token += 1

                num_seq += 1
                tensor = torch.IntTensor(article_wids)
                # print( dict.string_complete(tensor) )
                ds.add_item(tensor)

            print('| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}'.format(
                lang, input_file, num_seq, num_token,
                100 * num_unk_token / num_token, dict.unk_word if hasattr(dict, 'unk_word') else '<no_unk_word>'))

        ds.finalize(dataset_dest_path(output_prefix, lang, 'idx'))

    def make_dataset(input_prefix, output_prefix, lang):
        if args.output_format == 'binary':
            make_binary_dataset(input_prefix, output_prefix, lang)
        elif args.output_format == 'raw':
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + '.{}-{}'.format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)

    def make_all(lang):
        if args.trainpref:
            make_dataset(args.trainpref, 'train', lang)
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(',')):
                outprefix = 'valid{}'.format(k) if k > 0 else 'valid'
                make_dataset(validpref, outprefix, lang)
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(',')):
                outprefix = 'test{}'.format(k) if k > 0 else 'test'
                make_dataset(testpref, outprefix, lang)

    make_all(args.source_lang)
    if target:
        make_all(args.target_lang)

    print('| Wrote preprocessed data to {}'.format(args.destdir))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    if args.only_source:
        import os
        os.system('cp dict.summary.txt {}/dict.{}.txt'.format(args.destdir, args.target_lang))
