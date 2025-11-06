#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Functions for Model Loading
====================================
This module provides a factory function to instantiate different
neural network architectures for medical image classification.

The get_model() function supports multiple pre-built architectures
from the MONAI framework as well as custom implementations.

Usage:
    >>> from utils import get_model
    >>> model = get_model('sfcn', spatial_dims=3, in_channels=1, out_channels=2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import monai
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
import config_file as cfg
# Import custom SFCN implementation
from SFCN_Class import SFCNModelMONAIClassification, SFCN


def get_model(model_name, spatial_dims=3, in_channels=1, out_channels=2):
    """
    Factory function to instantiate different neural network architectures.
    
    This function provides a unified interface for loading various 3D CNN
    architectures commonly used in medical image analysis. It supports both
    MONAI's pre-built models and custom implementations.
    
    Args:
        model_name (str): Name of the model architecture to instantiate.
                         Options: 'densenet', 'resnet', 'efficientnet', 
                                 'highresnet', 'senet', 'vit', 'sfcn_monai', 
                                 'sfcn', 'cnn3d'
        spatial_dims (int): Number of spatial dimensions (2 for 2D, 3 for 3D).
                           Default: 3 (for volumetric MRI data)
        in_channels (int): Number of input channels. 
                          Default: 1 (for grayscale MRI)
        out_channels (int): Number of output classes.
                           Default: 2 (for binary classification: Autism vs Control)
    
    Returns:
        torch.nn.Module: Instantiated neural network model ready for training
    
    Raises:
        ValueError: If model_name is not recognized
    
    Examples:
        >>> # Load SFCN model
        >>> model = get_model('sfcn')
        
        >>> # Load DenseNet with custom parameters
        >>> model = get_model('densenet', spatial_dims=3, in_channels=1, out_channels=3)
    
    Note:
        All models are returned in their initialized state (random weights).
        You need to train them or load pre-trained weights separately.
    """
    
    # ==================== DenseNet ====================
    # Densely Connected Convolutional Networks
    # Good for: Feature reuse, gradient flow
    # Reference: Huang et al. (2017) - CVPR
    if model_name == 'densenet':
        return monai.networks.nets.DenseNet(
            spatial_dims=spatial_dims,      # 3D convolutions
            in_channels=in_channels,        # Input channels
            out_channels=out_channels       # Number of classes
            )
    
    # ==================== ResNet ====================
    # Deep Residual Learning for Image Recognition
    # Good for: Deep networks, skip connections prevent vanishing gradients
    # Reference: He et al. (2016) - CVPR
    elif model_name == 'resnet':
        return monai.networks.nets.ResNet(
            block = 'basic',                        # Use basic residual blocks
            layers = [2, 2, 2, 2],                  # 4 residual layers with 2 blocks each
            block_inplanes = [64, 128, 256, 512],   # Number of channels in each layer
            spatial_dims=spatial_dims,              # 3D operations
            n_input_channels=in_channels            # Input channels
            # Note: out_channels not needed for ResNet in MONAI
            )

    # ==================== EfficientNet ====================
    # Compound Scaling for Neural Networks
    # Good for: Efficiency, balanced depth/width/resolution scaling
    # Reference: Tan & Le (2019) - ICML
    elif model_name == 'efficientnet':
        return monai.networks.nets.EfficientNetBN(
            model_name="efficientnet-b0",   # Base model (smallest)
            spatial_dims=spatial_dims,      # 3D convolutions
            in_channels=in_channels,        # Input channels
            num_classes=out_channels        # Output classes
            )
    
    # ==================== HighResNet ====================
    # High-Resolution 3D Network
    # Good for: Preserving spatial details, medical imaging
    # Reference: Li et al. (2017) - MICCAI
    elif model_name == 'highresnet':
        return monai.networks.nets.HighResNet(
            spatial_dims=spatial_dims,      # 3D operations
            in_channels=in_channels,        # Input channels
            out_channels=out_channels       # Number of classes
            )
    
    # ==================== SENet ====================
    # Squeeze-and-Excitation Networks
    # Good for: Channel-wise feature recalibration, attention mechanism
    # Reference: Hu et al. (2018) - CVPR
    elif model_name == 'senet':
        return monai.networks.nets.SENet(
            spatial_dims=spatial_dims,          # 3D operations
            in_channels=in_channels,            # Input channels
            block = 'se_bottleneck',            # Use SE bottleneck blocks
            layers = [3, 4, 6, 3],              # Number of blocks per layer (SENet-154)
            groups = 64,                        # Number of groups for grouped convolutions
            reduction = 16,                     # Reduction ratio for SE blocks
            num_classes=out_channels            # Output classes
            )
    
    # ==================== Vision Transformer (ViT) ====================
    # Transformer architecture adapted for images
    # Good for: Long-range dependencies, global context
    # Reference: Dosovitskiy et al. (2021) - ICLR
    elif model_name == 'vit':
        return monai.networks.nets.ViT(
            spatial_dims=spatial_dims,      # 3D patches
            # Image dimensions from config file
            img_size=(cfg.params['imagex'], cfg.params['imagey'], cfg.params['imagez']),
            in_channels=in_channels,        # Input channels
            num_classes=out_channels,       # Output classes
            pos_embed='conv',               # Convolutional positional embeddings
            patch_size=(16, 16, 16)         # Size of 3D patches
            )
    
    # ==================== SFCN (MONAI-based) ====================
    # Custom SFCN implementation using MONAI building blocks
    elif model_name == 'sfcn_monai':
        return SFCNModelMONAIClassification(
            spatial_dims = spatial_dims,    # 3D operations
            in_channels = in_channels,      # Input channels
            out_channels = out_channels     # Output classes
            )
    
    # ==================== SFCN (Custom Implementation) ====================
    # Simple Fully Convolutional Network - Primary model for this project
    # Good for: Lightweight, interpretable, good baseline performance
    # This is the main model used in the autism classification pipeline
    elif model_name == 'sfcn':
        return SFCN()  # Custom implementation with progressive dropout
    
    # ==================== CNN3D (Custom) ====================
    # Custom 3D CNN architecture
    # Note: Requires CNN3D_Class.py to be present
    elif model_name == 'cnn3d':
        from CNN3D_Class import CNN3D  # Import only when needed
        return CNN3D(
            cfg.params,                     # Pass config parameters
            spatial_dims = spatial_dims,    # 3D operations
            in_channels = in_channels,      # Input channels
            out_channels = out_channels     # Output classes
            )
    
    # ==================== Error Handling ====================
    # Raise error if model name is not recognized
    else:
        raise ValueError(
            f"Unsupported model_name: {model_name}. "
            f"Supported models are: densenet, resnet, efficientnet, "
            f"highresnet, senet, vit, sfcn_monai, sfcn, cnn3d."
        )
