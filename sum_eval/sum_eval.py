
import os, sys
import re
import shutil
from collections import defaultdict
from .pyrouge_plus import get_rouge, get_rouge_multi_ref
import tqdm

SENT_SEP = '<S_SEP>'
SUM_SEP = '<SUM_SEP>'
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


def load_dict(result_file):
    word2idx = {}
    idx2word = {}
    fin = open(result_file, encoding=ENCODE)
    N = int(fin.readline())
    for i in range(N):
        line = fin.readline()
        fields = line.strip().split('\t')
        word = fields[0]
        idx = int(fields[1])
        word2idx[word] = idx
        idx2word[idx] = word

    print(word2idx)
    print(idx2word)

    return {'word2idx':word2idx, 'idx2word':idx2word}


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


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)


def extract_summary_without_rerank(article, true_labels, opts):
    pred_summary = []
    backup = []
    for sent_id, lbl in enumerate(true_labels):
        if lbl == 'T':
            pred_summary.append(article[sent_id])
            if len(pred_summary) >= opts['topk']:
                break
        elif lbl == 'F':
            backup.append(article[sent_id])
    if len(pred_summary) < opts['topk']:
        for sent in backup:
            pred_summary.append(sent)
            if len(pred_summary) >= opts['topk']:
                break

    return pred_summary


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


def extract_summary_with_rerank(article, pred_labels, pred_dist, pred_score, opts, vocab, nsent_budget, nword_budget, trigram_block):
    def get_value(dist, lbl):
        word2idx = vocab['word2idx']
        if lbl in word2idx:
            idx = word2idx[lbl]
            return dist[idx]
        return 0.0

    def get_weight(dist):
        true_prob = get_value(dist, 'T')
        maybe_prob = get_value(dist, 'M')
        return true_prob + 0.5 * maybe_prob if opts['with_m'] else true_prob

    # print pred_labels
    # print pred_dist
    if pred_score:
        scores = lmap(float, pred_score)
    elif pred_dist:
        scores = lmap(get_weight, pred_dist)
    else:
        raise NotImplementedError
    # print scores
    idxs = list(range(len(scores)))
    idxs.sort(key=lambda i: scores[i], reverse=True)
    # print idxs

    pred_sums = []
    trigram_set = set()
    if nsent_budget is None and nword_budget is None:
        for i in idxs:
            if trigram_block:
                cur_trigrams = sent2trigram(article[i])
                if in_trigram_set(trigram_set, cur_trigrams):
                    continue
                else:
                    add_trigram(trigram_set, cur_trigrams)
            pred_sums.append(article[i])
            if len(pred_sums) >= opts['topk']:
                break
    elif nsent_budget:
        for i in idxs:
            pred_sums.append(article[i])
            if len(pred_sums) >= nsent_budget:
                break
    else: # nword_budget is not None
        budget = 0
        for i in idxs:
            pred_sums.append(article[i])
            budget += len( article[i].strip().split() )
            if budget >= nword_budget:
                break
    return pred_sums


def generate_summary(article, summary, true_labels, pred_labels, pred_dist, pred_score, opts, vocab, nsent_budget, nword_budget, trigram_block):
    if opts['eval_type'] == 'lead':
        return article[0:min(len(article), opts['topk'])]
    elif opts['eval_type'] == 'gold':
        return extract_summary_without_rerank(article, true_labels, opts)
    elif opts['eval_type'] == 'predict':
        if not opts['rerank']:
            return extract_summary_without_rerank(article, pred_labels, opts)
        else:
            return extract_summary_with_rerank(article, pred_labels, pred_dist, pred_score, opts, vocab, nsent_budget, nword_budget, trigram_block)
    else:
        raise Exception('eval_type %s not recognized!'%opts['eval_type'])


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


def evaluate_extractive(result_file, article_file, summary_file,
                        entity_map_file=None, out_rouge_file=None,
                        cmd='-a -c 95 -m -n 4 -w 1.2', length=-1,
                        eval_type='lead', # options: lead, gold, predict
                        topk=3, rerank=False, with_m=False, add_full_stop=True,
                        nsent_budget_file=None, nword_budget_file=None, # buget files
                        multi_ref=False, # multiple references
                        trigram_block=False, # block repeated trigrams during sentence selection
                        ):
    articles = doc2sents(article_file, add_full_stop)
    summaries = doc2sents(summary_file, add_full_stop)

    if entity_map_file is not None:
        entity_maps = load_entity(entity_map_file)
        articles = deanonymize(entity_maps, articles)
        summaries = deanonymize(entity_maps, summaries)

    vocab = load_dict(result_file)
    results = load_result(result_file)
    ndoc = len(articles)
    opts = {'eval_type':eval_type, 'topk':topk,
            'rerank':rerank, 'with_m':with_m}

    outdir = os.path.join(os.path.dirname(out_rouge_file), '__tmp__rouge.%d' % os.getpid())
    mkdir(outdir)
    print('sum output dir: ', outdir)
    sysdir = os.path.join(outdir, 'sys')
    refdir = os.path.join(outdir, 'ref')
    mkdir(sysdir)
    mkdir(refdir)

    # handle budgets
    if nsent_budget_file:
        nsent_budgets = list( map(int, open(nsent_budget_file, encoding='utf8')) )
        assert ndoc == len(nsent_budgets)
    if nword_budget_file:
        nword_budgets = list( map(int, open(nword_budget_file, encoding='utf8')) )
        assert ndoc == len(nword_budgets)
    print(ndoc, len(results['PredictedLabels']))
    assert ndoc == len(results['PredictedLabels'])
    pred_scores = results.get('Score', None)
    try:
        for docid in tqdm.tqdm(range(ndoc)):
            article = articles[docid]
            summary = summaries[docid]
            true_labels = results['TrueLabels'][docid]
            pred_labels = results['PredictedLabels'][docid]
            try:
                pred_dist = results['PredictedDistri'][docid]
            except IndexError:
                pred_dist = []
            # pred_dist = [1 for _ in true_labels]
            pred_score = pred_scores[docid] if pred_scores is not None else None
            nsent_budget = nsent_budgets[docid] if nsent_budget_file else None
            nword_budget = nword_budgets[docid] if nword_budget_file else None
            try:
                pred_summary = generate_summary(article, summary, true_labels,
                            pred_labels, pred_dist, pred_score, opts, vocab, nsent_budget, nword_budget, trigram_block)
            except IndexError as e:
                if(len(article)==2):
                    pred_summary = article
                else:
                    raise e
            except NotImplementedError as e:
                print(e)
                shutil.rmtree(outdir)

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
            output_dict, output = get_rouge(sysdir, refdir, cmd=cmd, length=length)
        else:
            output_dict, output = get_rouge_multi_ref(sysdir, refdir, cmd=cmd, length=length)
    finally:
        shutil.rmtree(outdir)
        pass
    if out_rouge_file is not None:
        with open(out_rouge_file, 'w', encoding=ENCODE) as fout:
            fout.write(output)

    return output_dict, output

import multiprocessing, time
import multiprocessing.pool

class MultiProcSumEval(object):

    def __init__(self, ncpu):
        self.pool = multiprocessing.Pool(ncpu)
        # self.pool = multiprocessing.pool.ThreadPool(ncpu)
        self.ncpu = ncpu

    def add_eval_job(self, article_file, summary_file, entity_map_file,
                      result_file, out_rouge_file, length=-1, eval_type='lead',
                      topk=3, rerank=False, with_m=False,
                      cmd='-a -c 95 -m -n 4 -w 1.2', add_full_stop=False,
                      nsent_budget_file=None, nword_budget_file=None,
                      multi_ref=False,
                      trigram_block=False):
        outfile = out_rouge_file
        if not out_rouge_file.endswith('.rouge'):
            outfile += '.%d.top%d' % (length, topk)
            if rerank:
                outfile += '.rerank'
                if with_m:
                    outfile += '.with_m'
            if add_full_stop:
                outfile += '.add_full_stop'
            outfile += '.rouge'
        entity_map_file = None
        if self.ncpu == 1:
            evaluate_extractive(result_file, article_file, summary_file,
                                entity_map_file, outfile, cmd,
                                length, eval_type, topk, rerank, with_m, add_full_stop,
                                nsent_budget_file, nword_budget_file, multi_ref, trigram_block)
        else:
            self.pool.apply_async(evaluate_extractive, args=(
                                  result_file, article_file, summary_file,
                                  entity_map_file, outfile, cmd,
                                  length, eval_type, topk, rerank, with_m, add_full_stop,
                                  nsent_budget_file, nword_budget_file, multi_ref, trigram_block))

    def join(self):
        self.pool.close()
        print('join')
        self.pool.join()
