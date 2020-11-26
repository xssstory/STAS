import sys
import os
import argparse
from multiprocessing import Process

#pool = multiprocessing.Pool(20)

parser = argparse.ArgumentParser()

parser.add_argument('--model-dir', default='released_model/cnndm_model/ensemble_result/ensemble')
parser.add_argument('--gold-valid', default='data/cnndm/valid.summary')
parser.add_argument('--gold-test', default='data/cnndm/test.summary')

args = parser.parse_args()

def evaluate(gold_sum, gene_sum, out_rouge_file):
    os.system('python sum_eval_v2.py --gold_sum={} --gene_sum={} --out_rouge_file={}'.format(
        gold_sum, gene_sum, out_rouge_file))
ps = []
for split in ['valid', 'test']:
    sub_dir = os.path.join(args.model_dir, split)
    gold_sum = args.gold_valid if split == 'valid' else args.gold_test
    for gene_sum_file in os.listdir(sub_dir):
        gene_sum = os.path.join(sub_dir, gene_sum_file)
        out_rouge_file = gene_sum + '.rouge'
        p = Process(target=evaluate, args=(gold_sum, gene_sum, out_rouge_file))
        #pool.apply_async(evaluate, args=(gold_sum, gene_sum, out_rouge_file))
        p.start()
        ps.append(p)
for p in ps:
    p.join()
