
from .constants import Constants
from .dictionary import Dictionary
from .flexible_dictionary import FlexibleDictionary
import os, torch

class PointerFlexibleDictionary(FlexibleDictionary):

    def __init__(self, n_normal, specialTokens=[('PAD', '<pad>'), ('EOS', '</s>'), ('UNK', '<unk>')], luaHeritage=False):
        if luaHeritage:
            specialTokens = [('luaHeritage', '<luaHeritage>')] + specialTokens
        self.constants = Constants(specialTokens)

        self.symbols = []
        self.count = []
        self.indices = {}

        for idx in range(n_normal):
            self.add_symbol(str(idx))

        for attr, tok in specialTokens:
            self.add_symbol(tok)

        self.nspecial = len(self.symbols)

        if 'EOS' in self.constants.__dict__:
            self.eos_word = self.constants.EOS_WORD
            self.eos_index = self.index(self.eos_word)

        if 'PAD' in self.constants.__dict__:
            self.pad_word = self.constants.PAD_WORD
            self.pad_index = self.index(self.pad_word)

        if 'UNK' in self.constants.__dict__:
            self.unk_word = self.constants.UNK_WORD
            self.unk_index = self.index(self.unk_word)

        if 'BOS' in self.constants.__dict__:
            self.bos_word = self.constants.BOS_WORD
            self.bos_index = self.index(self.bos_word)
