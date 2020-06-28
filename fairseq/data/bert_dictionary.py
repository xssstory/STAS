
from .constants import Constants
from .dictionary import Dictionary
import os, torch
from collections import OrderedDict

class BertDictionary(Dictionary):

    def __init__(self):
        self.symbols = []
        self.count = []
        self.indices = OrderedDict()

    def find_special_tokens(self):
        '''[PAD] [UNK] [CLS] [SEP] [MASK]'''
        self.pad_word = '[PAD]'
        self.pad_index = self.indices[self.pad_word]
        self.unk_word = '[UNK]'
        self.unk_index = self.indices[self.unk_word]
        self.cls_word = '[CLS]'
        self.cls_index = self.indices[self.cls_word]
        self.sep_word = '[SEP]'
        self.sep_index = self.indices[self.sep_word]
        self.mask_word = '[MASK]'
        self.mask_index = self.indices[self.mask_word]

    def add_special_token(self, tok):
        if tok in self.indices:
            return self.indices[tok]
        else:
            for i in range(len(self.symbols)):
                if self.symbols[i].startswith('[unused'):
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
    def load(cls, f, ignore_utf_errors=False):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0>
        <symbol1>
        ...
        ```
        """
        if isinstance(f, str):
            try:
                if not ignore_utf_errors:
                    with open(f, 'r', encoding='utf-8') as fd:
                        return cls.load(fd)
                else:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as fd:
                        return cls.load(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except Exception:
                raise Exception("Incorrect encoding detected in {}, please "
                                "rebuild the dataset".format(f))

        d = cls()

        for line in f:
            word = line.strip()
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
        for symbol in self.symbols:
            print('{}'.format(symbol), file=f)

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        if hasattr(self, 'eos_index'):
            t[-1] = self.eos()
        return t
