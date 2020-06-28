
dataset=pac_nyt_orig
roberta_model=roberta-base
batch_size=4
update_freq=4
# batch_size = 4 x 4 x 4
predict_arch=pointer_net
warmup=5000
lr=0.00004,0.00004,0.0004
pointer_net_attn_type=perceptron
max_sent_length=40
mask_weight=0.5
perm_weight=0.5

DATA_DIR=data
datadir=$DATA_DIR/nyt/roberta_base_orig_bin
raw_datadir=$DATA_DIR/nyt
raw_valid=$raw_datadir/valid
raw_test=$raw_datadir/test
target=summary

modeldir=nyt_model_dir
mkdir $modeldir

cp $0 $modeldir

pip install pytorch-transformers==1.1.0 --user

python train.py $datadir \
--save-interval=1 \
--max-roberta-position=512 \
--task=perm_and_predict_mask --source-lang=article --target-lang=$target \
--raw-valid=$raw_valid --raw-test=$raw_test \
-a=perm_and_predict_mask_medium --optimizer=adam --lr=$lr \
--dropout=0.1 --max-sentences=$batch_size \
--max-sent-length=$max_sent_length \
--min-lr='-1' --lr-scheduler=multi_lr_inverse_sqrt --weight-decay=0.01 \
--criterion=sents_perm_with_mask \
--warmup-updates=$warmup --warmup-init-lr='0' \
--adam-betas='(0.9,0.999)' --save-dir=$modeldir \
--max-epoch=200 \
--update-freq=$update_freq \
--relu-dropout=0.1 --attention-dropout=0.1 \
--valid-subset=valid,test \
--max-sentences-valid=2 \
--sentence-transformer-arch=bert \
--roberta-model=${roberta_model} \
--predict-arch=$predict_arch \
--pointer-net-attn-type=$pointer_net_attn_type \
--multiple-lr \
--shuffle-prob=0 \
--mask-other-sents=False \
--bert-no-decay \
--ncpu-eval=1 \
--fp16-scale-window=1024 \
--ddp-backend=no_c10d \
--fp16 \
--masked-sent-loss-weight=$mask_weight --sent-perm-weight=$perm_weight \
--log-interval=50 2>&1 | tee $modeldir/log.txt
