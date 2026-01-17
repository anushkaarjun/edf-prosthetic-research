#!/usr/bin/env python3
"""
Improved CNN-LSTM model with increased depth and width.
Based on the original CNN-LSTM but with:
- Deeper CNN layers (4 conv blocks vs 2)
- Wider channels (32→64→128→256 vs 32→64)
- Deeper LSTM (2 layers, 256 hidden units vs 1 layer, 128)
- Deeper classifier (3 FC layers vs 1)
- Better regularization (progressive dropout)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedCNNLSTM(nn.Module):
    """
    Improved CNN-LSTM model for 3-class motion classification.
    Features:
    - Increased depth: 4 convolutional blocks (vs 2 original)
    - Increased width: 32→64→128→256 channels (vs 32→64)
    - Deeper LSTM: 2 layers, 256 hidden units (vs 1 layer, 128)
    - Deeper classifier: 3 FC layers (vs 1)
    - Progressive dropout: Higher in early layers, lower later
    """
    
    def __init__(self, n_channels, n_classes=3, dropout=0.5):
        super().__init__()
        
        # Block 1: First temporal-spatial convolution (wider)
        self.conv1 = nn.Conv2d(1, 64, (1, 5), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(64)
        
        # Block 2: Spatial convolution (wider channels)
        self.conv2 = nn.Conv2d(64, 128, (n_channels, 5), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(128)
        
        # Block 3: Deeper temporal convolution (even wider)
        self.conv3 = nn.Conv2d(128, 128, (1, 5), padding=(0, 2))
        self.bn3 = nn.BatchNorm2d(128)
        
        # Block 4: Final spatial-temporal convolution (widest)
        self.conv4 = nn.Conv2d(128, 256, (1, 5), padding=(0, 2))
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layers
        self.pool1 = nn.MaxPool2d((1, 2))
        self.pool2 = nn.MaxPool2d((1, 2))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # Adaptive pooling for flexibility
        
        # Progressive dropout (higher in early layers)
        self.dropout1 = nn.Dropout(dropout)  # 0.5
        self.dropout2 = nn.Dropout(dropout * 0.7)  # 0.35
        self.dropout3 = nn.Dropout(dropout * 0.5)  # 0.25
        self.dropout4 = nn.Dropout(dropout * 0.3)  # 0.15
        
        # Deeper LSTM: 2 layers, 256 hidden units (vs 1 layer, 128)
        self.lstm = nn.LSTM(
            input_size=256,  # Wider input from conv4
            hidden_size=256,  # Wider hidden state (vs 128)
            num_layers=2,  # Deeper (vs 1 layer)
            batch_first=True,
            dropout=0.2,  # LSTM dropout
            bidirectional=False  # Can enable bidirectional for even better performance
        )
        
        # Deeper classifier: 3 FC layers (vs 1)
        self.fc1 = nn.Linear(256, 512)  # Wider first layer
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, n_classes)
        self.dropout_fc = nn.Dropout(dropout * 0.2)  # 0.1
        
    def forward(self, x):
        """
        Forward pass through improved CNN-LSTM.
        Input: (batch, 1, n_channels, n_samples) e.g., (32, 1, 64, 320)
        Output: (batch, n_classes)
        """
        # Block 1: First convolution (wider)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        # Block 2: Spatial convolution (wider)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout2(x)
        
        # Block 3: Deeper temporal convolution
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        
        # Block 4: Final convolution (widest)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.adaptive_pool(x)  # Adaptive pooling
        x = self.dropout4(x)
        
        # Reshape for LSTM: (batch, time, features)
        b, f, ch, t = x.shape
        # After pooling, ch should be 256, f should be 1
        x = x.view(b, f * ch, t).permute(0, 2, 1)  # (batch, time, features)
        
        # Deeper LSTM (2 layers, 256 hidden)
        x, _ = self.lstm(x)
        # Use last time step
        x = x[:, -1, :]  # (batch, 256)
        
        # Deeper classifier (3 FC layers)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)
        x = self.fc3(x)
        
        return x


# Keep same configuration constants for compatibility
SLIDING_WINDOW = 320     # 2 sec at 160Hz
WINDOW_STEP = 80         # 75% overlap
N_CLASSES = 3

CLASS_NAMES = [
    'Open Left Fist',
    'Open Right Fist',
    'Close Fists'
]
