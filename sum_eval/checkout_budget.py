
import os, sys, argparse

def show_budget(art_file, sum_file, label_file):
    cnt = 0
    longer_cnt = 0
    eq_cnt = 0
    for art_line, sum_line, label_line in zip( open(art_file, encoding='utf8'), open(sum_file, encoding='utf8'), open(label_file, encoding='utf8') ):
        article = art_line.strip().split(' <S_SEP> ')
        nwords = len( sum_line.strip().split() )
        if nwords == 0:
            nwords = 1
        act_nwords = 0
        labels = label_line.strip().split()
        length = 0
        for sent, label in zip(article, labels):
            if label == 'T':
                length += len( sent.strip().split() )
        cnt += 1
        if length > nwords:
            longer_cnt += 1
        elif length == nwords:
            eq_cnt += 1

    print('total {}, longer {}, equal {}'.format(cnt, longer_cnt, eq_cnt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('art_file')
    parser.add_argument('sum_file')
    parser.add_argument('label_file')
    args = parser.parse_args()

    show_budget(args.art_file, args.sum_file, args.label_file)
