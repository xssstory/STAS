
from sum_eval import MultiProcSumEval
import argparse, os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', default='/disk/scratch/XingxingZhang/summarization/experiments/extract_baseline_rouge_lbl/base_ms30/run.save.1layer.sh.models/3.test.txt')
    parser.add_argument('--article', default='/disk/scratch/XingxingZhang/summarization/dataset/cnn_dailymail_rouge_lbl_sseg/test.article')
    parser.add_argument('--summary', default='/disk/scratch/XingxingZhang/summarization/dataset/cnn_dailymail_rouge_lbl_sseg/test.summary')
    parser.add_argument('--entity_map')
    parser.add_argument('--out_dir', default='.')
    parser.add_argument('--add_full_stop', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    result_file = args.result
    article_file = args.article
    summary_file = args.summary
    entity_map_file = args.entity_map

    mp_sum_eval = MultiProcSumEval(5)

    outfile = os.path.join(args.out_dir, 'lead3')
    mp_sum_eval.add_eval_job(article_file, summary_file, entity_map_file, result_file, outfile, eval_type='lead', topk=3, add_full_stop=args.add_full_stop)
    outfile = os.path.join(args.out_dir, 'gold3')
    mp_sum_eval.add_eval_job(article_file, summary_file, entity_map_file, result_file, outfile, eval_type='gold', topk=3, add_full_stop=args.add_full_stop)
    outfile = os.path.join(args.out_dir, 'predict')
    mp_sum_eval.add_eval_job(article_file, summary_file, entity_map_file, result_file, outfile, eval_type='predict', topk=3, add_full_stop=args.add_full_stop)
    mp_sum_eval.add_eval_job(article_file, summary_file, entity_map_file, result_file, outfile, eval_type='predict', topk=3, rerank=True, add_full_stop=args.add_full_stop)
    mp_sum_eval.add_eval_job(article_file, summary_file, entity_map_file, result_file, outfile, eval_type='predict', topk=3, rerank=True, with_m=True, add_full_stop=args.add_full_stop)

    mp_sum_eval.join()
    print('All jobs done!')
