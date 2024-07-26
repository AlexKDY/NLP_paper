import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn import init
import random 

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=4, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        self._init_weights()

    def _init_weights(self):
        for param in self.lstm.parameters():
            nn.init.uniform_(param, -0.08, 0.08)
        initrange = 1.0 / self.hidden_size
        init.uniform_(self.embedding.weight.data, -initrange, initrange)


    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(next(self.parameters()).device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size).to(next(self.parameters()).device))
    

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=4, dropout = 0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)

        self._init_weights()

    def _init_weights(self):
        for param in self.lstm.parameters():
            nn.init.uniform_(param, -0.08, 0.08)
        initrange = 1.0 / self.hidden_size
        init.uniform_(self.embedding.weight.data, -initrange, initrange)

    def forward(self, input, hidden):
        embedded = self.embedding(input).unsqueeze(1)
        output, hidden = self.lstm(embedded, hidden)
        output = self.softmax(self.out(output.squeeze(1)))

        return output, hidden

class Seq2Seq(nn.Module):
    def __init___(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input, target, tearcher_forcing_ratio = 0.5):
        batch_size = input.size(0)
        max_len = target.size(1)
        prob_size = self.decoder.out.out_features
        decoder_outputs = torch.zeros(batch_size, max_len, prob_size).to(self.device)
        hidden = self.encoder.init_hidden(batch_size)

        encoder_outputs, hidden = self.encoder(input, hidden)
        decoder_input = target[:, 0]

        for seq_idx in range(1, max_len):
            output, hidden = self.decoder(decoder_input, hidden)
            decoder_outputs[:, seq_idx, :] = output
            teacher_force = random.random() < tearcher_forcing_ratio
            pred = output.argmax(1)
            decoder_input = target[:, seq_idx] if teacher_force else pred

        return decoder_outputs








