import torch
import torch.nn as nn
import math


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


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PoseTransformer(nn.Module):
    """
    Transformer encoder that直接输出全身关节序列。
    """

    def __init__(
        self,
        input_size: int = 72,
        full_output_size: int = 72,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.full_head = nn.Linear(d_model, full_output_size)

    def forward(self, x: torch.Tensor, key_padding_mask=None):
        """
        Args:
            x: (B, T, input_size)
            key_padding_mask: (B, T) bool, True 表示 padding
        Returns:
            full_pred: (B, T, full_output_size)
        """
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        full_pred = self.full_head(h)
        return full_pred


class PoseTransformerCond(nn.Module):
    """
    Transformer encoder -> 先预测叶节点，再将叶节点映射回 d_model 融合，输出全身关节。
    """

    def __init__(
        self,
        input_size: int = 72,
        leaf_output_size: int = 18,
        full_output_size: int = 72,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.leaf_head = nn.Linear(d_model, leaf_output_size)
        self.leaf_to_model = nn.Linear(leaf_output_size, d_model)
        self.full_head = nn.Linear(d_model, full_output_size)

    def forward(self, x: torch.Tensor, key_padding_mask=None):
        """
        Args:
            x: (B, T, input_size)
            key_padding_mask: (B, T) bool, True 表示 padding
        Returns:
            leaf_pred: (B, T, leaf_output_size)
            full_pred: (B, T, full_output_size)
        """
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        leaf_pred = self.leaf_head(h)
        leaf_feat = self.leaf_to_model(leaf_pred)
        h_cond = h + leaf_feat

        full_pred = self.full_head(h_cond)
        return leaf_pred, full_pred
