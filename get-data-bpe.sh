datadir=data/nyt/

train=$datadir/train
valid=$datadir/valid
test=$datadir/test

dataoutdir=$datadir/roberta_base_bin

echo $dataoutdir

mkdir -vp $dataoutdir

srcdict=roberta-base-vocab.json

python preprocess_sum_roberta.py --source-lang article --target-lang summary \
    --trainpref $train --validpref $valid --testpref $test \
    --srcdict $srcdict \
    --destdir $dataoutdir \
    --only-source
