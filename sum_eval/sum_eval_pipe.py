
import os, sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import argparse
from sum_eval import MultiProcSumEval
from sum_eval.summarize_rouge import summarize_rouge
import re


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ncpu', type=int, default=4)
    parser.add_argument('-topk', type=int, default=3)
    parser.add_argument('-raw_valid', default='/disk/scratch/XingxingZhang/summarization/dataset/cnn_dailymail_all_in_one/validation')
    parser.add_argument('-raw_test', default='/disk/scratch/XingxingZhang/summarization/dataset/cnn_dailymail_all_in_one/test')
    parser.add_argument('-model_dir', default='/disk/scratch/XingxingZhang/summarization/experiments/extract_baseline/with_eval_multi_ovocab_nopad/run.save.1layer.sh.models')
    parser.add_argument('-add_full_stop', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    opts = get_args()
    eval_pool = MultiProcSumEval(opts.ncpu)
    valid_pool_params = dict(article_file=opts.raw_valid + '.article',
                              summary_file=opts.raw_valid + '.summary',
                              entity_map_file=opts.raw_valid + '.entity_map',
                              length=-1, eval_type='predict',
                              topk=opts.topk, rerank=False, with_m=False,
                              cmd='-a -c 95 -m -n 4 -w 1.2',
                              add_full_stop=args.add_full_stop)

    test_pool_params = dict(article_file=opts.raw_test + '.article',
                              summary_file=opts.raw_test + '.summary',
                              entity_map_file=opts.raw_test + '.entity_map',
                              length=-1, eval_type='predict',
                              topk=opts.topk, rerank=False, with_m=False,
                              cmd='-a -c 95 -m -n 4 -w 1.2',
                              add_full_stop=args.add_full_stop)

    def make_params(default_dict, result_file, out_rouge_file, rerank=False, with_m=False):
        para_dict = dict(default_dict)
        para_dict['result_file'] = result_file
        para_dict['out_rouge_file'] = out_rouge_file
        para_dict['rerank'] = rerank
        para_dict['with_m'] = with_m
        return para_dict

    def parse_fname(fname):
        m = re.search('(\\d+).(valid).txt', fname)
        if m:
            return int(m.group(1)), m.group(2)
        return None, None

    job_info = []
    for f in os.listdir(opts.model_dir):
        epoch, split = parse_fname(f)
        if epoch is not None:
            job_info.append( (epoch, opts.model_dir) )
    job_info.sort(key=lambda x: x[0])
    for epoch, model_dir in job_info:
        print(epoch, model_dir)
        valid_result_file = '%s/%d.valid.txt' % (model_dir, epoch)
        valid_out_file = '%s/%d.valid' % (model_dir, epoch)
        eval_pool.add_eval_job(**make_params(valid_pool_params, valid_result_file, valid_out_file, False, False))
        eval_pool.add_eval_job(**make_params(valid_pool_params, valid_result_file, valid_out_file, True, False))
        test_result_file = '%s/%d.test.txt' % (model_dir, epoch)
        test_out_file = '%s/%d.test' % (model_dir, epoch)
        eval_pool.add_eval_job(**make_params(test_pool_params, test_result_file, test_out_file, False, False))
        eval_pool.add_eval_job(**make_params(test_pool_params, test_result_file, test_out_file, True, False))

    eval_pool.join()
    print('evaluation done!')
    summarize_rouge(opts.model_dir)
