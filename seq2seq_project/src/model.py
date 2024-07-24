import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn import init

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=4, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)

    def _init_weights(self):
        for param in self.lstm.parameters():
            nn.init.uniform_(param, -0.08, 0.08)
        initrange = 1.0 / self.hidden_size
        init.uniform_(self.embedding.weight.data, -initrange, initrange)


    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden
    

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

    def forward(self, input, hidden):
        embedded = self.embedding(input).unsqueeze(1)
        output, hidden = self.lstm(embedded, hidden)
        output = self.softmax(self.out(output.squeeze(1)))

        return output, hidden
    

class Seq2Seq(nn.Module):
    def __init___(self):
        super(Seq2Seq, self).__init__()
    
