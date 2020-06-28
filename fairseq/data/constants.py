
class Constants(object):

    def __init__(self, specialTokens=[('PAD', '<blank>'), ('UNK', '<unk>'),
                       ('BOS', '<s>'), ('EOS', '</s>')],
                       specialTokenTemplate='<%s>'):
        assert isinstance(specialTokens, list), 'specialTokens must be a list'
        self.specialTokenTemplate = specialTokenTemplate
        self.specials = []
        self.specialNames = []
        for item in specialTokens:
            if isinstance(item, tuple):
                self.add(item[0], item[1])
            elif isinstance(item, str):
                self.add(item)
            else:
                assert True, 'invalid item type in specialTokens'

    def add(self, tokenName, tokenString=None):
        if tokenString is None:
            tokenString = self.specialTokenTemplate % tokenName
        self.__dict__[tokenName + '_WORD'] = tokenString
        nextID = len(self.specials)
        self.specials.append(tokenString)
        self.specialNames.append(tokenName)
        self.__dict__[tokenName] = nextID
