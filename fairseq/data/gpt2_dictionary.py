
from .constants import Constants
from .dictionary import Dictionary
import os, torch
from collections import OrderedDict

class GPT2Dictionary(Dictionary):

    def __init__(self):
        self.symbols = []
        self.count = []
        self.indices = OrderedDict()

    def find_special_tokens(self):
        '''
        <s> 0
        <pad> 1
        </s> 2
        <unk> 3
        ...
        <mask> 50264
        '''
        self.pad_word = '<pad>'
        self.pad_index = self.indices[self.pad_word]
        self.unk_word = '<unk>'
        self.unk_index = self.indices[self.unk_word]
        self.cls_word = '<s>'
        self.cls_index = self.indices[self.cls_word]
        self.sep_word = '</s>'
        self.sep_index = self.indices[self.sep_word]
        self.mask_word = '<mask>'
        self.mask_index = self.indices[self.mask_word]
        # begin of sentence and end of sentence
        self.bos_word = '<s>'
        self.bos_index = self.indices[self.bos_word]
        self.eos_word = '</s>'
        self.eos_index = self.indices[self.eos_word]

    def add_special_token(self, tok):
        if tok in self.indices:
            return self.indices[tok]
        else:
            for i in range(len(self.symbols)):
                if self.symbols[i].startswith('[unused') or self.symbols[i].startswith('madeupword'):
                    orig_tok = self.symbols[i]
                    self.symbols[i] = tok
                    del self.indices[orig_tok]
                    self.indices[tok] = i

                    return i

            print('No space for token "{}"'.format(tok))
            return None


    def finalize(self, threshold=-1, nwords=-1, padding_factor=1):
        super(BertDictionary, self).finalize(threshold, nwords, padding_factor)

    @classmethod
    def load(cls, vocab_file):
        """Loads the dictionary from a gpt-2 json file:
        'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
        'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
        'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-vocab.json",

        Actually these json files are the same!
        """

        d = cls()
        import json
        word2idx = json.load(open(vocab_file, encoding="utf-8"))
        max_idx = -1
        idx2word = OrderedDict()
        for word, idx in word2idx.items():
            max_idx = max(max_idx, idx)
            idx2word[idx] = word

        unused_word_id = 0
        for i in range(max_idx + 1):
            if i in idx2word:
                word = idx2word[i]
            else:
                word = '[unused{}]'.format(unused_word_id)
                unused_word_id += 1
            count = 1 # no count information available
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(count)

        d.find_special_tokens()

        return d

    def save(self, f):
        """Stores dictionary into a text file"""
        if isinstance(f, str):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, 'w', encoding='utf-8') as fd:
                return self.save(fd)
        import json
        json.dump(self.indices, f)

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        if hasattr(self, 'eos_index'):
            t[-1] = self.eos()
        return t
