from TaskModelling.ResponseFunctions import ResponseFunctions
import os
from Seq2Seq.Encoder import EncoderRNN
from Seq2Seq.Decoder import LuongAttnDecoderRNN
from Seq2Seq.Attention import Attn
import torch
import torch.nn as nn
import unicodedata
import re
from Seq2Seq.GreedySearchDecoder import GreedySearchDecoder
from IPython.display import Markdown
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
class MakeConv(object):
    def __init__(self, embedding_path, task_path, device):
        self.functions = ResponseFunctions(embedding_path, task_path)
        pathf = 'C:/Users/mkork/Desktop/dat'
        save_dir = os.path.join("data", "save")
        corpus_name = 'ConvAIandFriends'
        self.attn_model = 'dot'
        model_name = 'ConvAIandFriends'
        self.hidden_size = 600
        self.encoder_n_layers = 2
        self.decoder_n_layers = 2
        self.dropout = 0.1
        batch_size = 64
        checkpoint_iter = 20000
        self.EOS = 2
        self.SOS = 1
        self.PAD = 0
        self.device=device

        self.MAX_LENGTH = 10

        self.loadFilename = os.path.join(pathf, save_dir, model_name, corpus_name,
                                    '{}-{}_{}'.format(self.encoder_n_layers, self.decoder_n_layers, self.hidden_size),
                                    '{}_checkpoint.tar'.format(checkpoint_iter))



    def load_weights(self):
        if self.loadFilename:
            # If loading on same machine the model was trained on
            checkpoint = torch.load(self.loadFilename)
            # If loading a model trained on GPU to CPU
            # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
            encoder_sd = checkpoint['en']
            decoder_sd = checkpoint['de']
            encoder_optimizer_sd = checkpoint['en_opt']
            decoder_optimizer_sd = checkpoint['de_opt']
            embedding_sd = checkpoint['embedding']
            self.voc = checkpoint['voc_dict']
            xx = checkpoint['embedding']

        tuple_list = list(xx.items())
        num_words = tuple_list[0][1].shape[0]
        self.embedding = nn.Embedding(num_words, self.hidden_size)
        if self.loadFilename:
            self.embedding.load_state_dict(xx)
        encoder = EncoderRNN(self.hidden_size, self.embedding, self.encoder_n_layers, self.dropout)
        decoder = LuongAttnDecoderRNN(self.attn_model, self.embedding, self.hidden_size, num_words, self.decoder_n_layers, self.dropout)
        if self.loadFilename:
            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.encoder.eval()
        self.decoder.eval()

        self.searcher = GreedySearchDecoder(encoder, decoder, self.device)

    def check_words(self,inputs):
        inputs = inputs.split(' ')
        for inputa in inputs:
            if inputa in self.functions.Related_word:
                return True
        return False



    def unicodeToAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def indexesFromSentence(self,voc, sentence):
        return [voc['word2idx'][word] for word in sentence.split(' ')] + [self.EOS]

    def sentenceOperation(self,s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s



    def evaluate(self,sentence):
        ### Format input sentence as a batch
        # words -> indexes
        indexes_batch = [self.indexesFromSentence(self.voc, sentence)]
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(self.device)
        lengths = lengths.to(self.device)
        # Decode sentence with searcher
        tokens, scores = self.searcher(input_batch, lengths, self.MAX_LENGTH)
        # indexes -> words
        decoded_words = [self.voc['idx2word'][token.item()] for token in tokens]

        return decoded_words

    def startConv(self):
        self.load_weights()
        input_sentence = ''
        while (1):
            try:
                # Get input sentence
                input_sentence = input('> ')
                res = self.check_words(input_sentence)
                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit': break

                if res:
                    if (input_sentence != 'bye'):
                        user_i = input_sentence
                        ques = self.functions.embeddings.check_distance(user_i)
                        quest, response = self.functions.sort_response(ques, user_i)
                        responsek = response
                        display(Markdown(f'<span style="color: red">Do you want to ask this:{quest}?</span>'))
                    else:
                        response = self.End_conv()
                        Stop = True
                    if len(response.split('\n')) > 1:
                        for resp_p in response.split('\n'):
                            display(Markdown(f'<span style="color: blue">{resp_p}</span>'))
                    else:
                        display(Markdown(f'<span style="color: blue">{response}</span>'))
                        # Normalize sentence
                else:
                    input_sentence = self.sentenceOperation(input_sentence)
                    # Evaluate sentence
                    output_words = self.evaluate(input_sentence)
                    # Format and print response sentence
                    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                    print('Bot:', ' '.join(output_words))

            except KeyError:
                print("Error: Encountered unknown word.")

