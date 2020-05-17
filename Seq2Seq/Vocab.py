

class Vocab(object):
    def __init__(self, name):
        self.PAD = 0
        self.SOS = 1
        self.EOS = 2
        self.name = name
        self.word2idx = {}
        self.wordcount = {}
        self.idx2word = {self.PAD: 'PAD', self.EOS: 'EOS', self.SOS: 'SOS'}
        self.numberofWord = 3

    def add_Sentence(self, sentence):
        for word in sentence.split(' '):
            self.check_Word(word)

    def check_Word(self, word):
        if word not in self.word2idx:
            self.wordcount[word] = 1
            self.word2idx[word] = self.numberofWord
            self.idx2word[self.numberofWord] = word
            self.numberofWord += 1
        else:
            self.wordcount[word] += 1