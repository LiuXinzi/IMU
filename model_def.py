import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden,
                 n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.input_proj = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            input_size=n_hidden,
            hidden_size=n_hidden,
            num_layers=n_rnn_layer,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.output_proj = nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)

    def forward(self, x, h=None):
        y = torch.relu(self.input_proj(self.dropout(x)))
        y, h = self.rnn(y, h)
        output = self.output_proj(y)
        return output, h


class PoseLSTM(nn.Module):

    def __init__(self,
                 input_size=72,
                 leaf_output_size=18,
                 full_output_size=72,
                 leaf_hidden_size=256,
                 full_hidden_size=64):
        super().__init__()
        self.net1 = RNN(input_size, leaf_output_size, leaf_hidden_size)
        self.net2 = RNN(input_size + leaf_output_size, full_output_size, full_hidden_size)

    def forward(self, x):
        leaf_seq, _ = self.net1(x)
        if self.training:
            noise = torch.randn_like(leaf_seq) * 0.04
            leaf_seq_noisy = leaf_seq + noise
        else:
            leaf_seq_noisy = leaf_seq
        net2_input = torch.cat((x, leaf_seq_noisy), dim=-1)
        full_seq, _ = self.net2(net2_input)
        return leaf_seq, full_seq
