#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration File for SFCN-Based Autism Classification
========================================================
This module contains all hyperparameters for training the 3D Convolutional
Neural Network on structural MRI data from the ABIDE dataset.

Usage:
    Import this module to access training parameters:
    >>> import config_file as cfg
    >>> batch_size = cfg.params['batch_size']
"""

# Dictionary containing all training and model hyperparameters
params = {
    # ==================== Training Parameters ====================
    'batch_size': 2,           # Number of 3D MRI volumes per batch
                               # Note: Keep small (1-2) due to high memory requirements of 3D data
    
    'lr': 0.00001,             # Learning rate for Adam optimizer (1e-5)
                               # Small learning rate for stable convergence
    
    'epochs': 130,             # Maximum number of training epochs
                               # Training may stop earlier due to early stopping
    
    'patience': 30,            # Number of epochs to wait for improvement before early stopping
                               # Higher patience allows more time for model to improve
    
    # ==================== Image Dimensions ====================
    # Target dimensions for resizing input MRI volumes
    'imagex': 161,             # Width (sagittal plane)
    'imagey': 201,             # Height (coronal plane)  
    'imagez': 165,             # Depth (axial plane)
    # These dimensions are standard for preprocessed MRI data
    
    # ==================== Convolution Parameters ====================
    'kernel_size': (3, 3, 3),  # 3D convolutional kernel size
                               # (3,3,3) captures local spatial features
    
    'pool_size': (2, 2, 2),    # Max pooling size for downsampling
                               # Reduces spatial dimensions by half at each pooling layer
}
