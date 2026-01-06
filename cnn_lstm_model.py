#!/usr/bin/env python3
"""
CNN-LSTM model for motion classification.
Extracted from cnn-lstm.ipynb for use in API server.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    """CNN-LSTM model for 3-class motion classification."""
    
    def __init__(self, n_channels, n_classes=3, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, (1, 5), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, (n_channels, 5), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d((1, 2))
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        b, f, ch, t = x.shape
        x = x.view(b, f * ch, t).permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


# Model configuration constants
SLIDING_WINDOW = 320     # 2 sec at 160Hz
WINDOW_STEP = 80         # 75% overlap
N_CLASSES = 3

CLASS_NAMES = [
    'Open Left Fist',
    'Open Right Fist',
    'Close Fists'
]

