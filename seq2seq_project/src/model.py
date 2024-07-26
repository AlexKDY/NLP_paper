import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn import init
import random
from typing import List, Tuple

class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int = 4, dropout: float = 0.1) -> None:
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for param in self.lstm.parameters():
            nn.init.uniform_(param, -0.08, 0.08)
        initrange = 1.0 / self.hidden_size
        init.uniform_(self.embedding.weight.data, -initrange, initrange)

    def forward(self, input: torch.Tensor, input_lengths: List[int], hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(next(self.parameters()).device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size).to(next(self.parameters()).device))
    

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, n_layers: int = 4, dropout: float = 0.1) -> None:
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for param in self.lstm.parameters():
            nn.init.uniform_(param, -0.08, 0.08)
        initrange = 1.0 / self.hidden_size
        init.uniform_(self.embedding.weight.data, -initrange, initrange)

    def forward(self, input: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor], encoder_outputs: torch.Tensor, trg_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(input).unsqueeze(1)  # (batch_size, 1, hidden_size)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, trg_lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.softmax(self.out(output.squeeze(1)))  # (batch_size, output_size)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN, device: torch.device) -> None:
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input: torch.Tensor, src_lengths: List[int], target: torch.Tensor, trg_lengths: List[int], teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        batch_size = input.size(0)
        max_len = target.size(1)
        prob_size = self.decoder.out.out_features
        decoder_outputs = torch.zeros(batch_size, max_len, prob_size).to(self.device)
        hidden = self.encoder.init_hidden(batch_size)

        encoder_outputs, hidden = self.encoder(input, src_lengths, hidden)
        decoder_input = target[:, 0]

        for seq_idx in range(1, max_len):
            output, hidden = self.decoder(decoder_input, hidden, trg_lengths)
            decoder_outputs[:, seq_idx, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            pred = output.argmax(1)
            decoder_input = target[:, seq_idx] if teacher_force else pred

        return decoder_outputs
