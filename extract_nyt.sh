
roberta_type=base
dataset=nyt
max_sent_length=40
temperature=1,1.1,1.2,1.3
multi_graph=False
no_recovery=False
topk=3
modeldir=released_model/nyt_model/65_docker
pretrained_model=released_model/nyt_model/checkpoint65.pt

DATA_DIR=data

target=summary
case $dataset in
    "nyt")  
    datadir=$DATA_DIR/nyt/roberta_base_orig_bin
    raw_datadir=$DATA_DIR/nyt
    raw_valid=$raw_datadir/valid
    raw_test=$raw_datadir/test
    ;;
    "cnndm")
    datadir=$DATA_DIR/cnndm/cnndm_yangliu_label_bin
    raw_datadir=$DATA_DIR/cnndm
    raw_valid=$raw_datadir/validation
    raw_test=$raw_datadir/test
    target=label
    ;;
    *) echo "dataset does not support $dataset yet !"
    ;;
esac


mkdir $modeldir


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
--log-interval=100 2>&1 | tee $modeldir/log.txt
