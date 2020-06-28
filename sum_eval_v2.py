import os, sys
import argparse
import re
import multiprocessing
import shutil
from sum_eval.pyrouge_plus import get_rouge, get_rouge_multi_ref
from sum_eval.summarize_rouge import summarize_rouge
import multiprocessing
multiprocessing.set_start_method('spawn', True)

SENT_SEP = '<S_SEP>'
SUM_SEP = '<SUM_SEP>'
ENCODE = 'utf-8'

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)


def load_entity(entity_file):
    entity_maps = []
    for line in open(entity_file, encoding=ENCODE):
        entity_map = {}
        fields = line.strip().split('\t')
        for field in fields:
            pos = field.find(':')
            ne_name = field[0:pos].strip()
            ne = field[pos+1:].strip()
            entity_map[ne_name] = ne
        entity_maps.append(entity_map)

    return entity_maps


def line2sents(line, add_full_stop=True):
    sents = line.strip().split(SENT_SEP)
    # trick from Jianpeng Cheng
    if add_full_stop:
        return [sent.strip() + ' .' for sent in sents]
    return [sent.strip() for sent in sents]


def doc2sents(infile, add_full_stop=True):
    return [line2sents(line, add_full_stop) for line in open(infile, encoding=ENCODE)]


def deanonymize(entity_maps, articles):
    new_articles = []
    assert len(entity_maps) == len(articles)
    for i in range(len(entity_maps)):
        entity_map = entity_maps[i]
        article = articles[i]
        new_article = []
        for sent in article:
            words = sent.strip().split(' ')
            new_words = [entity_map.get(word, word) for word in words]
            new_sent = ' '.join(new_words)
            new_article.append(new_sent)
        new_articles.append(new_article)

    return new_articles


def write_multi_ref(refdir, docid, summary):
    def write(outfile, sents):
        with open(outfile, 'w', encoding=ENCODE) as fout:
            fout.write('\n'.join(sents))
            fout.write('\n')

    summary_line = (' ' + SENT_SEP + ' ').join( summary )
    summaries = summary_line.strip().split(' ' + SUM_SEP + ' ')
    # print(len(summaries))
    for sum_item in summaries:
        fds = sum_item.split('\t')
        assert len(fds) == 2
        label = fds[0]
        sum_sents = fds[1].strip().split(' ' + SENT_SEP + ' ')
        sum_sents = [sent.strip() for sent in sum_sents]
        fname = os.path.join(refdir, '{}.{}.gold'.format(label, docid))
        write(fname, sum_sents)


def evaluate_extractive(result_file, summary_file, add_full_stop,
                        entity_map_file=None, out_rouge_file=None,
                        cmd='-a -c 95 -m -n 4 -w 1.2', multi_ref=False, # multiple references
                        trigram_block=False, # block repeated trigrams during sentence selection
                        ):

    gold_summary = doc2sents(summary_file, add_full_stop)
    gene_summary = doc2sents(result_file, add_full_stop)

    if entity_map_file is not None:
        entity_maps = load_entity(entity_map_file)
        gold_summary = deanonymize(entity_maps, gold_summary)
        gene_summary = deanonymize(entity_maps, gene_summary)


    outdir = os.path.join( os.path.dirname(summary_file),'__tmp__rouge.%d' % os.getpid())
    print(outdir)
    mkdir(outdir)
    sysdir = os.path.join(outdir, 'sys')
    refdir = os.path.join(outdir, 'ref')
    mkdir(sysdir)
    mkdir(refdir)
    try:
        for docid, (summary, pred_summary) in enumerate(zip(gold_summary, gene_summary)):

            def write(outfile, sents):
                with open(outfile, 'w', encoding=ENCODE) as fout:
                    fout.write('\n'.join(sents))
                    fout.write('\n')

            write(os.path.join(sysdir, '%d.test' % docid), pred_summary)
            if not multi_ref:
                write(os.path.join(refdir, '%d.gold' % docid), summary)
            else:
                write_multi_ref(refdir, docid, summary)
        if not multi_ref:
            output_dict, output = get_rouge(sysdir, refdir, cmd=cmd, length=-1)
        else:
            output_dict, output = get_rouge_multi_ref(sysdir, refdir, cmd=cmd, length=1)
    finally:
        #pass
        shutil.rmtree(outdir)
    if out_rouge_file is not None:
        with open(out_rouge_file, 'w', encoding=ENCODE) as fout:
            fout.write(output)

    return output_dict, output


class MultiProcSumEval(object):

    def __init__(self, ncpu):
        self.pool = multiprocessing.Pool(ncpu)
        # self.pool = multiprocessing.pool.ThreadPool(ncpu)
        self.ncpu = ncpu

    def add_eval_job(self, summary_file, entity_map_file, result_file, out_rouge_file, add_full_stop,
                      cmd='-a -c 95 -m -n 4 -w 1.2', multi_ref=False, trigram_block=False):
        outfile = out_rouge_file

        if self.ncpu == 1:
            evaluate_extractive(result_file, summary_file, add_full_stop,
                                entity_map_file, outfile, cmd, multi_ref, trigram_block)
        else:
            self.pool.apply_async(evaluate_extractive, args=(
                                  result_file, summary_file, add_full_stop,
                                  entity_map_file, outfile, cmd, multi_ref, trigram_block))

    def join(self):
        self.pool.close()
        self.pool.join()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, default=1)
    parser.add_argument('--gold_summary', default='/home/v-shux/sum_data/cnndm_data/cnn_dailymail_qingyu_label_remove_none/test.summary')
    parser.add_argument('--gene_summary', default='../results/gene_summary.txt')
    parser.add_argument('--out_rouge_file', default='../results/gene_summart.rouge')
    parser.add_argument('--entity_map', default=None)
    parser.add_argument('--add_full_stop', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    eval_pool = MultiProcSumEval(args.ncpu)

    eval_pool.add_eval_job(
        summary_file=args.gold_summary, entity_map_file=args.entity_map, result_file=args.gene_summary,
        cmd='-a -c 95 -m -n 2 -w 1.2', out_rouge_file=args.out_rouge_file, add_full_stop=args.add_full_stop
    )
    
    
