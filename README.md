
STAS stands for **S**entence-level **T**ransformer based **A**ttentive **S**ummarization.

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

### Dataset

You can download CNN/DM and NYT datasets from <a link here> and  unzip the file, these files are organized as follows: 

```css
data 
    ├── cnndm 
        ├── cnndm_yangliu_label_bin 
            └── ... 
        ├── training.article 
        ├── ... 
        └── test.summary 
    └── nyt 
        ├── roberta_base_orig_bin
            └── ...
        ├── train.article
        ├── ...
        └── test.article
```

Where the `cnndm_yangliu_label_bin` and `roberta_base_orig_bin` are in binary format and are used for training, other files like `*.article` are in text format.

### Trained models

You can download our released model from <a link here>, the files are organized as follows:

```css
.
├── README.md
└── released_model
    ├── cnndm_model
        ├── checkpoint85.pt
        └── ensemble_result
            ├── pacsum
                ├── 61.text.txt
                └── 61.valid.txt
            └── stas
                ├── 13.text.txt
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
   ```

   The generated summaries and ROGUE socres will be stored in the `released_model/cnndm_model/ensemble_result/ensemble/test` and `released_model/cnndm_model/ensemble_result/ensemble/valid` .



