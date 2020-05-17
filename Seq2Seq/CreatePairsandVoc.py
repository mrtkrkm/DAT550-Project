import os
import json
import re
import unicodedata
from Seq2Seq.Vocab import  Vocab
import yaml
class CreatePairs(object):
    def __init__(self,DDIR, corpus, FutterancesPath, Max_length, convai_p, chatter_p):
        self.DDIR=DDIR
        self.path_f=os.path.join(DDIR, corpus, FutterancesPath)
        self.MAX_LENGTH=Max_length
        self.name='Friends-ConvAI-Chatter'
        self.textlist = []
        self.convaiPaths=convai_p
        self.chatterPaths = os.listdir(os.path.join(DDIR, chatter_p))
        self.chatter_p=chatter_p
        self.delimiter='\t'
        self.FriendscreateEncDecUt()
        self.ConvACreateEncDec()
        self.chatterCreateEncDec()

    def FriendscreateEncDecUt(self):
        utterances = []
        with open(self.path_f, 'r', encoding='utf-8') as p:
            for line in p:
                utterances.append(json.loads(line))
        idtext = {}
        for ut in utterances:
            idtext[ut['id']] = ut['text']
        text = ''

        for ut in utterances:
            text = ''
            if ut['reply-to'] != None:
                if ut['text'] != '' and idtext[ut['reply-to']] != '':
                    text += idtext[ut['reply-to']] + self.delimiter + ut['text']
                    self.textlist.append(text)

    def ConvACreateEncDec(self):
        paths = []
        for conv in self.convaiPaths:
            paths.append(os.path.join(self.DDIR, conv))
        utterancesc = []
        for path in paths:
            with open(path, 'r', encoding='utf-8') as p:
                for line in p:
                    utterancesc.append(json.loads(line))
        for utterance in utterancesc:
            for utter in utterance:
                if len(utter['dialog']) > 1:
                    oldsender = utter['dialog'][0]['sender_class']
                    for i in range(1, len(utter['dialog'])):
                        sender = utter['dialog'][i]['sender_class']
                        if sender != oldsender:
                            text = utter['dialog'][i - 1]['text'] + self.delimiter + utter['dialog'][i]['text']
                        oldsender = utter['dialog'][i]['sender_class']
                        self.textlist.append(text)

    def chatterCreateEncDec(self):
        for c_path in self.chatterPaths:
            file = os.path.join(self.DDIR, self.chatter_p, c_path)
            with open(file, 'r', encoding='utf-8') as c:
                utt = yaml.safe_load(c)
            if c_path == 'literature.yml':
                utt['conversations'][-2][1] = utt['conversations'][-2][1].replace('Tolkein', 'Tolkien')
            for convers in utt['conversations']:
                if type(convers[1]) == dict:
                    key = list(convers[1].keys())[0]
                    value = list(convers[1].values())[0]
                    text1 = convers[0] + self.delimiter + key
                    text2 = convers[0] + self.delimiter + value
                    self.textlist.append(text1)
                    self.textlist.append(text2)
                else:
                    text = convers[0] + self.delimiter + convers[1]
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
