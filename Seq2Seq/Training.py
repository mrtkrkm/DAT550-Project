import torch
import torch.nn as nn
from Seq2Seq.Encoder import EncoderRNN
from Seq2Seq.Decoder import LuongAttnDecoderRNN
import os
import random
from torch import optim
from Seq2Seq.GetTrainData import GetTrainData

class Training(object):
    def __init__(self,voc, device,hidden_size,encoder_n_layers,decoder_n_layers,dropout, batch_size):
        self.device=device
        model_name = 'cb_model'
        attn_model = 'dot'
        # attn_model = 'general'
        # attn_model = 'concat'
        self.hidden_size = hidden_size
        self.encoder_n_layers = encoder_n_layers
        self.decoder_n_layers = decoder_n_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.voc=voc
        self.PAD = 0
        self.SOS = 1
        self.EOS = 2

        # Set checkpoint to load from; set to None if starting from scratch
        loadFilename = None
        if loadFilename:
            self.checkpoint = torch.load(loadFilename)

            self.encoder_sd = self.checkpoint['en']
            self.decoder_sd = self.checkpoint['de']
            self.encoder_optimizer_sd = self.checkpoint['en_opt']
            self.decoder_optimizer_sd = self.checkpoint['de_opt']
            self.embedding_sd = self.checkpoint['embedding']
            voc.__dict__ = self.checkpoint['voc_dict']

        print('Building encoder and decoder ...')
        # Initialize word embeddings
        self.embedding = nn.Embedding(voc.numberofWord, hidden_size)
        if loadFilename:
            self.embedding.load_state_dict(self.embedding_sd)
        # Initialize encoder & decoder models
        self.encoder = EncoderRNN(hidden_size, self.embedding, encoder_n_layers, dropout)
        self.decoder = LuongAttnDecoderRNN(attn_model, self.embedding, hidden_size, voc.numberofWord, decoder_n_layers, dropout)
        if loadFilename:
            self.encoder.load_state_dict(self.encoder_sd)
            self.decoder.load_state_dict(self.decoder_sd)
        # Use appropriate device
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        print('Models built and ready to go!')

    def maskNLLLoss(self,inp, target, mask):
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(self.device)
        return loss, nTotal.item()

    def train(self,input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
              encoder_optimizer, decoder_optimizer, batch_size, clip):

        # Zero gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Set device options
        input_variable = input_variable.to(self.device)
        lengths = lengths.to(self.device)
        target_variable = target_variable.to(self.device)
        mask = mask.to(self.device)

        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[1 for _ in range(self.batch_size)]])
        decoder_input = decoder_input.to(self.device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, nTotal = self.maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(self.device)
                # Calculate and accumulate loss
                mask_loss, nTotal = self.maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

        # Adjust model weights
        encoder_optimizer.step()
        decoder_optimizer.step()

        return sum(print_losses) / n_totals

    def trainIters(self,model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding,
                   encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip,
                   corpus_name, loadFilename):

        # Load batches for each iteration
        gtd = GetTrainData(voc)
        self.all_lose=[]
        training_batches = [gtd.batch2TrainData([random.choice(pairs) for _ in range(batch_size)])
                            for _ in range(n_iteration)]

        # Initializations
        print('Initializing ...')
        start_iteration = 1
        print_loss = 0
        if loadFilename:
            start_iteration = self.checkpoint['iteration'] + 1

        # Training loop
        print("Training...")
        for iteration in range(start_iteration, n_iteration + 1):
            training_batch = training_batches[iteration - 1]
            # Extract fields from batch
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            # Run a training iteration with batch
            loss = self.train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, batch_size, clip)
            print_loss += loss

            # Print progress
            if iteration % print_every == 0:
                print_loss_avg = print_loss / print_every
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration,
                                                                                              iteration / n_iteration * 100,
                                                                                              print_loss_avg))
                self.all_lose.append(print_loss/print_every)
                print_loss = 0


            # Save checkpoint
            if (iteration % save_every == 0):
                directory = os.path.join(save_dir, model_name, corpus_name,
                                         '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, self.hidden_size))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save({
                    'iteration': iteration,
                    'en': encoder.state_dict(),
                    'de': decoder.state_dict(),
                    'en_opt': encoder_optimizer.state_dict(),
                    'de_opt': decoder_optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': voc.__dict__,
                    'embedding': embedding.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


    def fit(self, corpuss, model_name, voc, pairs, learning_rate, n_iteration, print_every, save_every, clip, corpus_name, loadFilename,teacher_forcing_ratio):
        # Configure training/optimization

        save_dir = os.path.join("data", "save")
        corpus_name = corpuss
        clip = clip
        self.teacher_forcing_ratio = teacher_forcing_ratio
        learning_rate = learning_rate
        decoder_learning_ratio = 5.0
        n_iteration = n_iteration
        print_every = print_every
        save_every = save_every
        checkpoint_iter = 10000


        # Ensure dropout layers are in train mode
        self.encoder.train()
        self.decoder.train()

        # Initialize optimizers
        print('Building optimizers ...')
        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
        if loadFilename:
            encoder_optimizer.load_state_dict(self.encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(self.decoder_optimizer_sd)

        # If you have cuda, configure cuda to call
        for state in encoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in decoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        # Run training iterations
        print("Starting Training!")
        self.trainIters(model_name, voc, pairs, self.encoder, self.decoder, encoder_optimizer, decoder_optimizer,
                   self.embedding, self.encoder_n_layers, self.decoder_n_layers, save_dir, n_iteration, self.batch_size,
                   print_every, save_every, clip, corpus_name, loadFilename)

        return self.all_lose
