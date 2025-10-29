import torch
import torch.nn as nn


class RNN(nn.Module):
    """
    A lightweight wrapper consisting of a linear input projection, an LSTM stack,
    and a linear output projection. Operates on sequences with shape
    (batch, seq_len, feature_dim) and returns the final-step prediction.
    """
    def __init__(self, n_input, n_output, n_hidden,
                 n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super().__init__()
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
        """
        Args:
            x: Tensor of shape (batch, seq_len, n_input)
        Returns:
            output: Tensor of shape (batch, n_output) for the last time step.
        """
        y = torch.relu(self.input_proj(self.dropout(x)))
        y = self.dropout(y)
        y, h = self.rnn(y, h)
        last_step = y[:, -1]
        output = self.output_proj(self.dropout(last_step))
        return output, h


class PoseLSTM(nn.Module):
    """
    Two-stage network:
      - net1 predicts IMU-adjacent leaf joint positions.
      - net2 consumes the original sequence concatenated with the leaf prediction
        to infer all body joints.
    """
    def __init__(self,
                 input_size=42,
                 leaf_output_size=18,
                 full_output_size=72,
                 leaf_hidden_size=256,
                 full_hidden_size=64):
        super().__init__()
        self.net1 = RNN(input_size, leaf_output_size, leaf_hidden_size)
        self.net2 = RNN(input_size + leaf_output_size, full_output_size, full_hidden_size)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_size)
        Returns:
            leaf_pred: (batch, leaf_output_size)
            full_pred: (batch, full_output_size)
        """
        leaf_pred, _ = self.net1(x)
        leaf_seq = leaf_pred.unsqueeze(1).expand(-1, x.size(1), -1)
        net2_input = torch.cat((x, leaf_seq), dim=-1)
        full_pred, _ = self.net2(net2_input)
        return leaf_pred, full_pred
