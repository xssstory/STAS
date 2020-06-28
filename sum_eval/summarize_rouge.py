
import os
from collections import defaultdict


def parse_rouge_file(infile):
    Rs = ['ROUGE-1 Average_F', 'ROUGE-2 Average_F', 'ROUGE-L Average_F']
    R1, R2, RL = 0.0, 0.0, 0.0
    def get_r(line):
        left = line.find(':')
        right = line.find('(')
        return float(line[left+1:right])
    for line in open(infile):
        if Rs[0] in line:
            R1 = get_r(line)
        elif Rs[1] in line:
            R2 = get_r(line)
        elif Rs[2] in line:
            RL = get_r(line)

    return R1, R2, RL


def split_fname(fname):
    fds = fname.split('.')
    assert len(fds) > 3
    assert fds[0].isdigit(), 'MUST be a number'
    return int(fds[0]), fds[1], '.'.join(fds[2:])


def summarize_rouge(indir):
    rlbl2perf = defaultdict(lambda : defaultdict(list))
    for f in os.listdir(indir):
        if f.endswith('.rouge'):
            epoch, split_lbl, rouge_lbl = split_fname(f)
            R1, R2, RL = parse_rouge_file(os.path.join(indir, f))
            rlbl2perf[rouge_lbl][split_lbl].append( (epoch, R1, R2, RL) )

    for rlbl, split2perf in rlbl2perf.items():
        print('*** %s ****' % rlbl)
        for split, perf in split2perf.items():
            perf.sort(key=lambda x: x[0])
        maxepoch = len(split2perf['valid'])
        assert maxepoch == len(split2perf['test']), 'MUST be the same length'
        best_epoch_r1, best_epoch_r2, best_epoch_rl = -1, -1, -1
        best_r1, best_r2, best_rl = 0, 0, 0
        for i in range(maxepoch):
            vepoch, vr1, vr2, vr3 = split2perf['valid'][i]
            tepoch, tr1, tr2, tr3 = split2perf['test'][i]
            assert vepoch == tepoch, 'should be the same!'
            print( 'epoch %d VALID R1 %f R2 %f RL %f, TEST R1 %f R2 %f RL %f' % (
                  vepoch, vr1, vr2, vr3, tr1, tr2, tr3) )
            if vr1 > best_r1:
                best_r1 = vr1
                best_epoch_r1 = vepoch
            if vr2 > best_r2:
                best_r2 = vr2
                best_epoch_r2 = vepoch
            if vr3 > best_rl:
                best_rl = vr3
                best_epoch_rl = vepoch

        def show(msg, best_epoch):
            for i in range(maxepoch):
                vepoch, vr1, vr2, vr3 = split2perf['valid'][i]
                tepoch, tr1, tr2, tr3 = split2perf['test'][i]
                if vepoch == best_epoch:
                    print('** best %s **'%msg)
                    vals = list(split2perf['valid'][i]) + list(split2perf['test'][i])[1:]
                    print('epoch %d VALID R1 %f R2 %f RL %f | TEST R1 %f R2 %f RL %f' % tuple(vals))

        print()
        show('R1', best_epoch_r1)
        show('R2', best_epoch_r2)
        show('RL', best_epoch_rl)

        print('\n\n\n')

        for split, perf in split2perf.items():
            r1 = map(lambda x: str(x[1]), perf)
            print( '%s_R%s = [%s]'%(split, '1', ','.join(r1)) )
            r2 = map(lambda x: str(x[2]), perf)
            print( '%s_R%s = [%s]'%(split, '2', ','.join(r2)) )
            rl = map(lambda x: str(x[3]), perf)
            print( '%s_R%s = [%s]'%(split, 'L', ','.join(rl)) )

        print('\n\n\n')
