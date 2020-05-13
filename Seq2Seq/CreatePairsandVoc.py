import os
import json
import re
import unicodedata
from Seq2Seq.Vocab import  Vocab
class CreatePairs(object):
    def __init__(self,DDIR, corpus, utterancesPath, Max_length):
        self.path=os.path.join(DDIR, corpus, utterancesPath)
        self.MAX_LENGTH=Max_length
        self.name=corpus
        self.createUtterances()
        self.setId()
        self.createEncDecUt()

    def createUtterances(self):
        self.utterances = []
        with open(self.path, 'r', encoding='utf-8') as p:
            for line in p:
                self.utterances.append(json.loads(line))

    def setId(self):
        self.idtext = {}
        for ut in self.utterances:
            self.idtext[ut['id']] = ut['text']

    def createEncDecUt(self):
        text = ''
        delimiter = '\t'
        self.textlist = []
        for ut in self.utterances:
            text = ''
            if ut['reply-to'] != None:
                if ut['text'] != '' and self.idtext[ut['reply-to']] != '':
                    text += self.idtext[ut['reply-to']] + delimiter + ut['text']
                    self.textlist.append(text)

    def unicodeToAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def sentenceOperation(self,s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s



    def read_voc(self,data):
        pairs = [[self.sentenceOperation(sentence) for sentence in sentences.split('\t')] for sentences in data]
        voc = Vocab(self.name)
        return pairs, voc

    # Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
    def filterPair(self,p):
        # Input sequences need to preserve the last word for EOS token
        return len(p[0].split(' ')) < self.MAX_LENGTH and len(p[1].split(' ')) < self.MAX_LENGTH

    # Filter pairs using filterPair condition
    def filterPairs(self,pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    def load_data(self):
        print('Starting')
        pairs, vocab = self.read_voc(self.textlist)
        print(f'Number of pairs is {len(pairs)}')
        pairs = self.filterPairs(pairs)
        print(f'After filters Number of pairs is {len(pairs)}')
        for pair in pairs:
            vocab.add_Sentence(pair[0])
            vocab.add_Sentence(pair[1])
        print("Counted words:", vocab.numberofWord)
        return vocab, pairs
