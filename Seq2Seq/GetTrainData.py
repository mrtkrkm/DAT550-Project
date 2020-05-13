import itertools
import torch
class GetTrainData(object):

    def __init__(self,voc):
        self.voc=voc
        self.EOS=2
        self.SOS=1
        self.PAD=0

    def indexesFromSentence(self, sentence):
        return [self.voc.word2idx[word] for word in sentence.split(' ')] + [self.EOS]

    def zeroPadding(self, l):
        return list(itertools.zip_longest(*l, fillvalue=self.PAD))

    def binaryMatrix(self,l):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == self.PAD:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    # Returns padded input sequence tensor and lengths
    def inputVar(self,l):
        indexes_batch = [self.indexesFromSentence( sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths

    # Returns padded target sequence tensor, padding mask, and max target length
    def outputVar(self, l):
        indexes_batch = [self.indexesFromSentence(sentence) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        mask = self.binaryMatrix(padList)
        mask = torch.ByteTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, max_target_len

    # Returns all items for a given batch of pairs
    def batch2TrainData(self, pair_batch):
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = self.inputVar(input_batch)
        output, mask, max_target_len = self.outputVar(output_batch)
        return inp, lengths, output, mask, max_target_len

