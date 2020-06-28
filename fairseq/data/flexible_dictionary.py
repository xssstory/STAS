
from .constants import Constants
from .dictionary import Dictionary
import os, torch

class FlexibleDictionary(Dictionary):

    def __init__(self, specialTokens=[('PAD', '<pad>'), ('EOS', '</s>'), ('UNK', '<unk>')], luaHeritage=False):
        if luaHeritage:
            specialTokens = [('luaHeritage', '<luaHeritage>')] + specialTokens
        self.constants = Constants(specialTokens)

        self.symbols = []
        self.count = []
        self.indices = {}

        for attr, tok in specialTokens:
            self.add_symbol(tok)

        self.nspecial = len(self.symbols)

        if 'PAD' in self.constants.__dict__:
            self.pad_word = self.constants.PAD_WORD
            self.pad_index = self.constants.PAD

        if 'EOS' in self.constants.__dict__:
            self.eos_word = self.constants.EOS_WORD
            self.eos_index = self.constants.EOS

        if 'UNK' in self.constants.__dict__:
            self.unk_word = self.constants.UNK_WORD
            self.unk_index = self.constants.UNK

    def finalize(self, threshold=-1, nwords=-1, padding_factor=1):
        super(FlexibleDictionary, self).finalize(threshold, nwords, padding_factor)

    @classmethod
    def load(cls, f, ignore_utf_errors=False, additonalSpecialTokens=None):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        if isinstance(f, str):
            try:
                if not ignore_utf_errors:
                    with open(f, 'r', encoding='utf-8') as fd:
                        return cls.load(fd, additonalSpecialTokens=additonalSpecialTokens)
                else:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as fd:
                        return cls.load(fd, additonalSpecialTokens=additonalSpecialTokens)
            except FileNotFoundError as fnfe:
                raise fnfe
            except Exception:
                raise Exception("Incorrect encoding detected in {}, please "
                                "rebuild the dataset".format(f))

        # load special tokens
        if additonalSpecialTokens is None:
            specialTokens = []
        else:
            specialTokens = additonalSpecialTokens

        if additonalSpecialTokens is None:
            for line in f:
                line = line.strip()
                if line == '###SP_TOKENS_START###':
                    continue
                if line == '###SP_TOKENS_END###':
                    break
                fds = line.split(' ')
                assert len(fds) == 2
                specialTokens.append( (fds[0], fds[1]) )
        d = cls(specialTokens)

        for line in f:
            idx = line.rfind(' ')
            word = line[:idx]
            count = int(line[idx+1:])
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(count)
        return d

    def save(self, f):
        """Stores dictionary into a text file"""
        """Special tokens included"""
        if isinstance(f, str):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, 'w', encoding='utf-8') as fd:
                return self.save(fd)
        # save special tokens
        print('###SP_TOKENS_START###', file=f)
        for spname, sptok in zip( self.constants.specialNames, self.constants.specials ):
            print('{} {}'.format(spname, sptok), file=f)
        print('###SP_TOKENS_END###', file=f)
        for symbol, count in zip(self.symbols[self.nspecial:], self.count[self.nspecial:]):
            print('{} {}'.format(symbol, count), file=f)

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        if hasattr(self, 'eos_index'):
            t[-1] = self.eos()
        return t
