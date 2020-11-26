
roberta_type=base
dataset=cnndm
max_sent_length=55
temperature=1
multi_graph=True
no_recovery=False
topk=3
recovery_thresh=0

DATA_DIR=data

target=summary
case $dataset in
    "nyt")  
    datadir=$DATA_DIR/nyt/roberta_base_bin
    raw_datadir=$DATA_DIR/nyt
    raw_valid=$raw_datadir/valid
    raw_test=$raw_datadir/test
    ;;
    "cnndm")
    datadir=$DATA_DIR/cnndm/roberta_base_bin
    raw_datadir=$DATA_DIR/cnndm
    raw_valid=$raw_datadir/valid
    raw_test=$raw_datadir/test
    target=summary
    ;;
    *) echo "dataset does not support $dataset yet !"
    ;;
esac

modeldir=released_model/cnndm_model/85
mkdir $modeldir

pretrained_model=released_model/cnndm_model/checkpoint85.pt
echo "checkpoint path: " $pretrained_model

pip install pytorch-transformers==1.1.0 --user

python rank_aml.py $datadir \
--task=extractive_summarization_recovery_dev --source-lang=article --target-lang=$target \
--raw-valid=$raw_valid --raw-test=$raw_test \
-a=extract_sum_recovery_pagerank_$roberta_type --optimizer=adam --lr=0.0001 \
--attn-type=attn_score \
--temperature=$temperature \
--multi-graph=$multi_graph \
--no-recovery=$no_recovery \
--dropout=0.1 --max-sentences=2 \
--min-lr='1e-09' --lr-scheduler=inverse_sqrt --weight-decay=0.01 \
--warmup-updates=10000 --warmup-init-lr='1e-08' \
--adam-betas='(0.9,0.999)' --save-dir=$modeldir \
--max-epoch=200 \
--max-sent-length=$max_sent_length \
--update-freq=32 \
--relu-dropout=0.1 --attention-dropout=0.1 \
--valid-subset=valid,test \
--max-sentences-valid=8 \
--roberta-model=roberta-$roberta_type \
--sentence-transformer-arch=bert \
--init-from-pretrained-doc-model=True \
--pretrained-doc-model-path=$pretrained_model \
--mask-other-sents=False \
--bert-no-decay \
--ncpu-eval=2 \
--topk=$topk \
--fp16 \
--recovery-thresh=$recovery_thresh \
--log-interval=100 2>&1 | tee $modeldir/log.txt
