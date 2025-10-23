import torch
import torch.nn as nn

class PoseLSTM(nn.Module):
    def __init__(self, input_size=42, hidden_size=512, output_size=72):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True,bidirectional=True,dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(h_last)