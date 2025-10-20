import torch
import torch.nn as nn

class PoseLSTM(nn.Module):
    def __init__(self, input_size=42, hidden_size=128, output_size=72):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        return self.fc(h_last)