#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Fully Convolutional Network (SFCN) for Autism Classification
====================================================================
This module implements the SFCN architecture with progressive dropout
for binary classification of autism from 3D structural MRI scans.

The network uses:
- 5 convolutional blocks with batch normalization
- Progressive dropout (increasing from 0.05 to 0.25)
- Global average pooling
- Lazy linear layer for classification

Architecture:
    Input (1×161×201×165) 
    → Conv3D(16) + BN + MaxPool + ReLU
    → Conv3D(32) + BN + MaxPool + ReLU + Dropout(0.05)
    → Conv3D(64) + BN + MaxPool + ReLU + Dropout(0.1)
    → Conv3D(128) + BN + MaxPool + ReLU + Dropout(0.15)
    → Conv3D(64, 1×1×1) + BN + ReLU + Dropout(0.2)
    → AvgPool + Dropout(0.25) + Flatten + Linear(2)

Reference:
    Peng et al. (2021) - Accurate brain age prediction with lightweight 
    deep neural networks. Medical Image Analysis, 68, 101871.
"""

import torch.nn as nn
import torch.nn.functional as F
import monai
from monai.networks.blocks import Convolution


class SFCN(nn.Module):
    """
    Simple Fully Convolutional Network with progressive dropout regularization.
    
    This architecture gradually increases dropout rate across layers to prevent
    overfitting while maintaining learning capacity in earlier layers.
    
    Attributes:
        conv1-5: 3D convolutional layers
        norm1-5: Batch normalization layers
        maxpool1-4: Max pooling layers for spatial downsampling
        dropout2-5, dropout_FINAL: Dropout layers with increasing rates
        avgpool: Global average pooling
        flatten: Flatten spatial dimensions
        classification: Lazy linear layer (auto-infers input size)
    """
    
    def __init__(self):
        """Initialize the SFCN architecture with 5 convolutional blocks."""
        super(SFCN, self).__init__()

        # ==================== Block 1: Input → 16 channels ====================
        # First layer extracts low-level features from grayscale MRI
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm1 = nn.BatchNorm3d(16)  # Normalize activations for stable training
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # Reduce spatial size by 2x

        # ==================== Block 2: 16 → 32 channels ====================
        # Increases feature complexity with light regularization
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm2 = nn.BatchNorm3d(32)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dropout2 = nn.Dropout(0.05)  # Light dropout (5%) to reduce overfitting

        # ==================== Block 3: 32 → 64 channels ====================
        # Mid-level features with moderate regularization
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm3 = nn.BatchNorm3d(64)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dropout3 = nn.Dropout(0.1)  # Moderate dropout (10%)

        # ==================== Block 4: 64 → 128 channels ====================
        # High-level feature extraction with stronger regularization
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm4 = nn.BatchNorm3d(128)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dropout4 = nn.Dropout(0.15)  # Stronger dropout (15%)

        # ==================== Block 5: 128 → 64 channels ====================
        # 1×1×1 convolution for channel reduction (dimensionality reduction)
        self.conv5 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.norm5 = nn.BatchNorm3d(64)
        self.dropout5 = nn.Dropout(0.2)  # High dropout (20%)

        # ==================== Block 6: Classification Head ====================
        # Global pooling and classification layers
        self.avgpool = nn.AvgPool3d(kernel_size=(2, 2, 2))  # Further spatial reduction
        self.flatten = nn.Flatten()  # Convert 3D feature maps to 1D vector
        self.dropout_FINAL = nn.Dropout(0.25)  # Highest dropout (25%) before classification
        self.classification = nn.LazyLinear(2)  # Binary classification (Autism vs Control)
                                                 # LazyLinear automatically determines input size on first forward pass

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, D, H, W)
                             where D=165, H=201, W=161 for standard MRI volumes
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 2)
                         Raw scores for each class (not probabilities)
        
        Note:
            Activation functions (ReLU) are applied after normalization and pooling.
            Dropout is applied after activation for regularization.
        """
        
        # ==================== Block 1 ====================
        # Input: (batch, 1, 161, 201, 165) → Output: (batch, 16, 80, 100, 82)
        x = F.relu(self.maxpool1(self.norm1(self.conv1(x))))

        # ==================== Block 2 ====================
        # Input: (batch, 16, 80, 100, 82) → Output: (batch, 32, 40, 50, 41)
        x = F.relu(self.maxpool2(self.norm2(self.conv2(x))))
        x = self.dropout2(x)  # Apply dropout after activation

        # ==================== Block 3 ====================
        # Input: (batch, 32, 40, 50, 41) → Output: (batch, 64, 20, 25, 20)
        x = F.relu(self.maxpool3(self.norm3(self.conv3(x))))
        x = self.dropout3(x)

        # ==================== Block 4 ====================
        # Input: (batch, 64, 20, 25, 20) → Output: (batch, 128, 10, 12, 10)
        x = F.relu(self.maxpool4(self.norm4(self.conv4(x))))
        x = self.dropout4(x)

        # ==================== Block 5 ====================
        # Input: (batch, 128, 10, 12, 10) → Output: (batch, 64, 10, 12, 10)
        # 1×1×1 convolution doesn't change spatial dimensions
        x = F.relu(self.norm5(self.conv5(x)))
        x = self.dropout5(x)

        # ==================== Block 6: Classification ====================
        # Average pooling: (batch, 64, 10, 12, 10) → (batch, 64, 5, 6, 5)
        x = self.avgpool(x)
        # Final dropout before flattening
        x = self.dropout_FINAL(x)
        # Flatten: (batch, 64, 5, 6, 5) → (batch, 64*5*6*5) = (batch, 9600)
        x = self.flatten(x)
        # Linear classification: (batch, 9600) → (batch, 2)
        x = self.classification(x)

        return x
