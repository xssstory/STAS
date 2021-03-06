
This is an implementation of the STAS (**S**entence-level **T**ransformer based **A**ttentive **S**ummarization) model described in  [Unsupervised Extractive Summarization by Pre-training Hierarchical Transformers](https://www.aclweb.org/anthology/2020.findings-emnlp.161/)

### Installation

You need to install python3 and following libararies

```bash
pip install pytorch==1.2
pip install pyrouge==0.1.3
pip install pytorch-transformers==1.1.0
python setup.py build
python setup.py develop

# For rouge-1.5.5.pl
sudo apt-get update
sudo apt-get install expat
sudo apt-get install libexpat-dev -y

sudo cpan install XML::Parser
sudo cpan install XML::Parser::PerlSAX
sudo cpan install XML::DOM

```

We also provide the `Dockerfile` we used to train and evaluate the model.


### Trained models

You can download our released models from [here](https://xingxingzhang.blob.core.windows.net/share/stas/model.zip), the files are organized as follows:

```css
.
├── README.md
└── released_model
    ├── cnndm_model
        ├── checkpoint85.pt
        └── ensemble_result
            ├── pacsum
                ├── 61.test.txt
                └── 61.valid.txt
            └── stas
                ├── 13.test.txt
                └── 13.valid.txt
    └── nyt_model
        ├── checkpoint65.pt
        └── ensemble_result
            ├── pacsum
                └── ...
            └── stas
                └── ...
```
We provide the sentence scores given by STAT and PASUM in the **ensemble_result**, you can combine the scores following  [Evaluation](#Evaluation) 3.

### data preprocess
You should split your data into train/validation/test subsets and get 6 files like train.article, train.summary, valid.article, valid.summary, test.article and test.summary,
and make sure that each line has one article/summary, the sentence in the article/summary is splited by "<S_SEP>". (we only use summaries for evaluation and test). Here is an example:
```
Apple 's first generation iPad launched on 3 April 2010 <S_SEP> In its five years on the market , 225 million devices have been sold <S_SEP> But larger smartphones and smart watches may herald its end <S_SEP> Sales for the iPad dropped 18 per cent in the final quarter of 2014
```

Then run the get-data-bpe.sh (modify the file path in the script accroding to you situation) and you will get a file folder for training and evaluating our model.


### Training

We provide the scripts for training on the CNN/DM and NYT datasets, We trained our models with 4 Nvidia Tesla V100GPUs and employed gradient accumulation technique.

```bash
bash train_cnndm.sh # For cnndm
bash train_nyt.sh # For nyt
```



### Evaluation

We also provide the steps to evaluate the models.

1. run the scripts to score the sentences

   ```bash
   bash extract_cnndm.sh # for cnndm
   bash extract_nyt.sh # for nyt
   ```

2. computing the ROUGE scores

   ```bash
   python sum_eval_pipe.py -raw_test=data/cnndm/test -raw_valid=data/cnndm/validation -model_dir=released_model/cnndm_model/85/ # for cnndm
   python sum_eval_pipe.py -raw_test=data/nyt/test -raw_valid=data/nyt/valid  -model_dir=released_model/nyt_model/65/ #for nyt
   ```


3. combine the scores given by STAS and PACSUM

   ```
   python ensemble.py
   python evaluate_ensemble.py
   # for nyt
   python ensemble.py --raw-valid=data/nyt/valid.article --raw-test=data/nyt/test.article --stas-dir=released_model/nyt_model/ensemble_result/stas/ --pacsum-dir=released_model/nyt_model/ensemble_result/pacsum/ --outdir=released_model/nyt_model/ensemble_result/ensenble/ --rerank=False

   ```

   The generated summaries and ROGUE socres will be stored in the `released_model/cnndm_model/ensemble_result/ensemble/test` and `released_model/cnndm_model/ensemble_result/ensemble/valid` .

### Citation
```
@inproceedings{xu-etal-2020-unsupervised,
    title = "Unsupervised Extractive Summarization by Pre-training Hierarchical Transformers",
    author = "Xu, Shusheng  and
      Zhang, Xingxing  and
      Wu, Yi  and
      Wei, Furu  and
      Zhou, Ming",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.161",
    pages = "1784--1795",
}
```


