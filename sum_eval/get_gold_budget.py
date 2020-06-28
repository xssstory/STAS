
import os, sys, argparse

def get_gold_sent_buget(label_file, buget_file):
    bugets = []
    with open(buget_file, 'w', encoding='utf8', buffering=1) as fout:
        for line in open(label_file, encoding='utf8'):
            nsent = line.count('T')
            if nsent == 0: nsent = 1
            fout.write( '{}\n'.format(nsent) )
            bugets.append(nsent)

    import numpy
    bugets = numpy.asarray(bugets)
    print( 'sentence buget: min {} max {} avg {} std {}'.format(bugets.min(), bugets.max(), bugets.mean(), bugets.std()) )

def get_gold_word_buget(sum_file, buget_file):
    bugets = []
    with open(buget_file, 'w', encoding='utf8', buffering=1) as fout:
        for line in open(sum_file, encoding='utf8'):
            nword = len(line.strip().split())
            if nword == 0: nsent = 1
            fout.write( '{}\n'.format(nword) )
            bugets.append(nword)

    import numpy
    bugets = numpy.asarray(bugets)
    print( 'word buget: min {} max {} avg {} std {}'.format(bugets.min(), bugets.max(), bugets.mean(), bugets.std()) )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gold_file')
    parser.add_argument('buget_file')
    parser.add_argument('--type', default='sentence')
    args = parser.parse_args()

    if args.type == 'sentence':
        get_gold_sent_buget(args.gold_file, args.buget_file)
    elif args.type == 'word':
        get_gold_word_buget(args.gold_file, args.buget_file)
