import numpy as np
import argparse
from collections import defaultdict
import os

SENT_SEP = '<S_SEP>'
ENCODE = 'utf8'


def lmap(func, lst): return list(map(func, lst))

def line2sents(line, add_full_stop=True):
    sents = line.strip().split(SENT_SEP)
    # trick from Jianpeng Cheng
    if add_full_stop:
        return [sent.strip() + ' .' for sent in sents]
    return [sent.strip() for sent in sents]


def doc2sents(infile, add_full_stop=True):
    return [line2sents(line, add_full_stop) for line in open(infile, encoding=ENCODE)]

def load_result(result_file):
    results = defaultdict(list)
    for line in open(result_file, encoding=ENCODE):
        line = line.strip()
        if ':\t' in line:
            fields = line.split(':\t')
            fields = line.split(':\t')
            assert len(fields) == 2, 'two fields'
            k = fields[0].replace(' ', '')
            v = fields[1]
            if k == 'PredictedDistri':
                vdists = [ lmap(float, dis.strip().split()) for dis in v.split('|') ]
                results[k].append(vdists)
            else:
                results[k].append(v.strip().split())

    return results

def sent2trigram(sent):
    ngram = 3
    words = sent.strip().split()
    trigrams = []
    for i in range(len(words) - ngram + 1):
        trigrams.append( ' '.join(words[i:i+ngram]) )

    return trigrams


def in_trigram_set(trigram_set, cur_trigrams):
    for trig in cur_trigrams:
        if trig in trigram_set:
            return True

    return False


def add_trigram(trigram_set, cur_trigrams):
    for trig in cur_trigrams:
        trigram_set.add(trig)

def get_pos(pred_score):

    scores = lmap(float, pred_score)

    # print scores
    idxs = list(range(len(scores)))
    idxs.sort(key=lambda i: scores[i], reverse=True)
    # print idxs

    pred_idx = []
    trigram_set = set()

    for i in idxs:
        pred_idx.append(i)

        if len(pred_idx) >= 3:
            break

    return pred_idx

def get_pos_with_rerank(article, pred_score):

    scores = lmap(float, pred_score)

    # print scores
    idxs = list(range(len(scores)))
    idxs.sort(key=lambda i: scores[i], reverse=True)
    # print idxs

    pred_idx = []
    trigram_set = set()

    for i in idxs:

        cur_trigrams = sent2trigram(article[i])
        if in_trigram_set(trigram_set, cur_trigrams):
            continue
        else:
            add_trigram(trigram_set, cur_trigrams)

        pred_idx.append(i)
        # shorten the summary
        # if i not in idxs[:opts['topk']]:
        #     break
        if len(pred_idx) >= 3:
            break

    return pred_idx


def ensemble_score(our_score, pacsum_score, lam1):

    def norm(score_list):
        score_list = lmap(float, score_list)
        score_list = [max(0, s) for s in score_list]
        score_sum = sum(score_list)
        if score_sum == 0:
            return score_list
        score_list = [s/score_sum for s in score_list]
        return score_list
    pacsum_score = pacsum_score[: len(our_score)]
    pacsum_score = norm(pacsum_score)
    our_score = norm(our_score)
    
    out_score = [a * lam1 + b * (1-lam1) for a, b in zip(our_score, pacsum_score)]
    
    return out_score

def extract_sum(article_file, results_our, results_pacsum, outdir, rerank):
    articles = doc2sents(article_file, False)
    our_scores = results_our['Score']
    pacsum_scores = results_pacsum['Score']
    assert len(pacsum_scores) == len(our_scores)
    for lam1 in range(11):
        # lam1 = 0.9 + lam1 / 100.
        lam1 = lam1 / 10.
        with open(os.path.join(outdir, 'our_{}.pac_{}.{}'.format(round(lam1, 3), round(1-lam1, 3), rerank)), 'w') as fout:
            for i, (article, our_score, pacsum_score) in enumerate(zip(articles, our_scores, pacsum_scores)):
                pred_score = ensemble_score(our_score, pacsum_score, lam1)

                article_pos = get_pos(pred_score) if not rerank else get_pos_with_rerank(article, pred_score)
                # article_pos.sort()
                summary = [article[pos] for pos in article_pos]
                for i, sent in enumerate(summary):
                    fout.write(sent) 
                    if i != len(summary) -1  :
                        fout.write("<S_SEP>")
                fout.write('\n')
                fout.flush()

def main(args):
    for split in ['valid', 'test']:
        result_list_stas = os.listdir(args.stas_dir)
        result_list_pacsum = os.listdir(args.pacsum_dir)
        assert len(result_list_pacsum) == 2
        assert len(result_list_stas) == 2
        for f in result_list_stas:
            if split in f:
                results_stas = load_result(os.path.join(args.stas_dir, f))
        for f in result_list_pacsum:
            if split in f:
                results_pacsum = load_result(os.path.join(args.pacsum_dir, f))
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        
        cur_outdir = os.path.join(args.outdir, split)
        if not os.path.exists(cur_outdir):
            os.makedirs(cur_outdir)
        article_file = args.raw_valid if split == 'valid' else args.raw_test 
        extract_sum(article_file, results_stas, results_pacsum, cur_outdir, eval(args.rerank))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw-valid', default='data/cnndm/valid.article')
    parser.add_argument('--raw-test', default='data/cnndm/test.article')
    
    parser.add_argument('--stas-dir', default='released_model/cnndm_model/ensemble_result/stas/')
    parser.add_argument('--pacsum-dir', default='released_model/cnndm_model/ensemble_result/pacsum/')
    
    parser.add_argument('--outdir', default='released_model/cnndm_model/ensemble_result/ensemble')
    parser.add_argument('--rerank', default='True')
    args = parser.parse_args()
    # args.rerank = eval(args.rerank)
    main(args)
