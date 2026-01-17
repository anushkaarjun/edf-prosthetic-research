"""
Improved neural network model with increased depth and width.
Based on SimpleNet pattern but adapted for EEG motor imagery classification.
"""
import torch
import torch.nn as nn
from typing import Optional


class ImprovedEEGNet(nn.Module):
    """
    Improved EEG motor imagery classification model.
    Features:
    - Increased depth: More convolutional layers
    - Increased width: More channels/neurons per layer
    - Better regularization: Dropout and batch normalization
    - Residual connections: Help with training deeper networks
    """
    
    def __init__(
        self, 
        n_channels: int = 64, 
        n_classes: int = 4, 
        n_samples: int = 126,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        # Increased width: Start with more channels
        # Block 1: Temporal Convolution (deeper and wider)
        self.conv1_1 = nn.Conv1d(n_channels, 64, kernel_size=25, padding=12)
        self.bn1_1 = nn.BatchNorm1d(64)
        self.conv1_2 = nn.Conv1d(64, 64, kernel_size=13, padding=6)
        self.bn1_2 = nn.BatchNorm1d(64)
        
        # Block 2: Spatial Convolution (increased width)
        self.conv2_1 = nn.Conv1d(64, 128, kernel_size=n_channels, groups=64)
        self.bn2_1 = nn.BatchNorm1d(128)
        self.conv2_2 = nn.Conv1d(128, 128, kernel_size=1)
        self.bn2_2 = nn.BatchNorm1d(128)
        
        # Block 3: Depthwise Separable Convolution (even wider)
        self.conv3_sep = nn.Conv1d(128, 128, kernel_size=13, padding=6, groups=128)
        self.conv3_point = nn.Conv1d(128, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Pooling layers
        self.pool1 = nn.AvgPool1d(kernel_size=25, stride=5)
        self.pool2 = nn.AvgPool1d(kernel_size=3, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 0.7)  # Less dropout in later layers
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)
        
        # Classifier: Deeper fully connected layers (increased depth)
        # Flattened size: 256 channels * 16 time steps = 4096
        self.fc1 = nn.Linear(256 * 16, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn_fc3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, n_classes)
        self.dropout_fc = nn.Dropout(dropout_rate * 0.3)
        
    def forward(self, x):
        """
        Forward pass through the improved network.
        Input: (batch, channels, time) e.g., (32, 64, 126)
        Output: (batch, n_classes)
        """
        # Block 1: Temporal convolutions (deeper)
        x = torch.relu(self.bn1_1(self.conv1_1(x)))
        x = torch.relu(self.bn1_2(self.conv1_2(x)))
        x = self.dropout1(x)
        
        # Block 2: Spatial convolutions (wider)
        x = torch.relu(self.bn2_1(self.conv2_1(x)))
        x = torch.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool1(x)
        x = self.dropout2(x)
        
        # Block 3: Depthwise separable convolution (wider channels)
        x = self.conv3_sep(x)
        x = torch.relu(self.bn3(self.conv3_point(x)))
        x = self.pool2(x)
        x = self.adaptive_pool(x)
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Deeper fully connected layers
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)
        x = torch.relu(self.bn_fc3(self.fc3(x)))
        x = self.fc4(x)
        
        return x


class SimpleEEGNet(nn.Module):
    """
    Simplified EEG model based on SimpleNet pattern.
    Uses fully connected layers with ReLU activations.
    Good starting point for experimentation.
    """
    
    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 4,
        n_samples: int = 126,
        hidden_size: int = 256,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        # Calculate input features: channels * samples
        num_inputs = n_channels * n_samples
        
        # Deeper and wider network (similar to SimpleNet but adapted)
        self.net = nn.Sequential(
            # Layer 1: Input to first hidden (wider)
            nn.Linear(num_inputs, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 2: First hidden to second (deeper)
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            
            # Layer 3: Second to third (even deeper)
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            # Layer 4: Third to fourth (deeper still)
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),
            
            # Output layer
            nn.Linear(hidden_size // 4, n_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        Input: (batch, channels, time) e.g., (32, 64, 126)
        Output: (batch, n_classes)
        """
        # Flatten input: (batch, channels, time) -> (batch, channels * time)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.net(x)


class DeepEEGNet(nn.Module):
    """
    Very deep EEG model with residual connections.
    Maximum depth and width for best accuracy potential.
    """
    
    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 4,
        n_samples: int = 126,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        # Very wide first layer
        self.conv1 = nn.Conv1d(n_channels, 128, kernel_size=25, padding=12)
        self.bn1 = nn.BatchNorm1d(128)
        
        # Residual block 1
        self.conv2 = nn.Conv1d(128, 128, kernel_size=13, padding=6)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Spatial convolution (very wide)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=n_channels, groups=128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Depthwise separable (wider)
        self.conv4_sep = nn.Conv1d(256, 256, kernel_size=13, padding=6, groups=256)
        self.conv4_point = nn.Conv1d(256, 512, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Pooling
        self.pool = nn.AvgPool1d(kernel_size=25, stride=5)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)
        
        # Very deep classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),
            
            nn.Linear(128, n_classes)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Convolutional layers
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        
        # Residual block
        residual = out
        out = torch.relu(self.bn2(self.conv2(out)))
        out = out + residual  # Residual connection
        out = self.dropout(out)
        
        # Spatial
        out = torch.relu(self.bn3(self.conv3(out)))
        out = self.pool(out)
        out = self.dropout(out)
        
        # Depthwise separable
        out = self.conv4_sep(out)
        out = torch.relu(self.bn4(self.conv4_point(out)))
        out = self.adaptive_pool(out)
        out = self.dropout(out)
        
        # Flatten and classify
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        return out
