#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20-Fold Cross-Validation Training Script for Autism Classification
===================================================================
This script trains the SFCN model using 20-fold cross-validation on
structural MRI data from the ABIDE dataset. Each fold is trained
independently with early stopping and checkpointing.

Features:
- 20-fold cross-validation for robust evaluation
- Reproducible training with fixed seed
- Early stopping based on validation loss
- Model checkpointing (saves best model per fold)
- Automatic accuracy logging and loss curve visualization

Directory Structure Expected:
    data_dir/
    ├── train_fold_1.csv
    ├── val_fold_1.csv
    ├── train_fold_2.csv
    ├── val_fold_2.csv
    └── ...

CSV Format:
    Columns: 'Link' (absolute path to .nii.gz), 'DX_GROUP' (0=Control, 1=Autism)

Usage:
    python my_training_model.py
    
    Note: Edit data_dir and outdir paths in train_fold() function before running
"""

import logging
import os
import sys
import random
import monai
import pickle
import torch
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import config_file as cfg
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from monai.data.utils import pad_list_data_collate
from tqdm import tqdm
from torchinfo import summary
from utils import get_model
from sklearn.model_selection import train_test_split
from monai.transforms import LoadImage, ScaleIntensityRanged, EnsureTyped, Compose, ScaleIntensity, EnsureChannelFirst, Resize
from monai.data import ImageDataset, DataLoader


def train_fold(fold_number, seed):
    """
    Train a single fold of the cross-validation.
    
    This function handles the complete training pipeline for one fold:
    - Sets up reproducibility
    - Loads data specific to this fold
    - Initializes model, optimizer, and loss
    - Trains with early stopping
    - Saves best model checkpoint
    - Logs accuracy and generates loss curves
    
    Args:
        fold_number (int): Current fold number (1-20)
        seed (int): Random seed for reproducibility
    
    Outputs:
        Creates a directory: {fold_number}_seed_{seed}/ containing:
        - Best_model_weights_{exp_name}.pth: Model state dict
        - Best_model_checkpoint_{exp_name}.pth: Full checkpoint (includes optimizer)
        - Best_model_checkpoint_{exp_name}.model: Complete model object
        - Best_model_checkpoint_{exp_name}.txt: Text file with best epoch info
        - accuracy_log_{exp_name}.csv: Per-epoch accuracy log
        - validation_metrics_{exp_name}.png: Loss curves plot
    """
    
    # ==================== Setup ====================
    # Define experiment name and output directory
    exp_name = f"fold_{fold_number}"
    outdir = f"/home/miplab/Desktop/CLEANING/{fold_number}_seed_{seed}"  # CHANGE THIS PATH
    os.makedirs(outdir, exist_ok=True)  # Create output directory if it doesn't exist
    model_weight_path = os.path.join(outdir, f'Best_model_weights_{exp_name}.pth')

    # ==================== Reproducibility ====================
    # Set all random seeds for deterministic behavior
    torch.manual_seed(seed)                          # PyTorch random seed
    torch.backends.cudnn.deterministic = True        # Force deterministic CUDA operations
    torch.backends.cudnn.benchmark = False           # Disable CUDA auto-tuner (for reproducibility)
    random.seed(seed)                                # Python random seed
    np.random.seed(seed)                             # NumPy random seed
    g = torch.Generator()                            # Generator for DataLoader
    g.manual_seed(seed)                              # Set generator seed

    # ==================== Data Loading ====================
    # Load training and validation CSV files for this specific fold
    data_dir = '/home/miplab/Desktop/CLEANING/20FOLD_TIPICAL_CV'  # CHANGE THIS PATH
    df_train = pd.read_csv(os.path.join(data_dir, f"train_fold_{fold_number}.csv")) 
    df_val = pd.read_csv(os.path.join(data_dir, f"val_fold_{fold_number}.csv"))
    
    # Extract file paths and labels as numpy arrays
    image_train = df_train['Link'].to_numpy()       # Paths to training MRI volumes
    label_train = df_train['DX_GROUP'].to_numpy()   # Training labels (0 or 1)
    image_val = df_val['Link'].to_numpy()           # Paths to validation MRI volumes
    label_val = df_val['DX_GROUP'].to_numpy()       # Validation labels (0 or 1)

    # ==================== Data Transforms ====================
    # Define preprocessing pipeline for training data
    train_transforms = Compose([
        ScaleIntensity(),       # Normalize voxel intensities to [0, 1]
        EnsureChannelFirst(),   # Add channel dimension if missing: (D,H,W) → (1,D,H,W)
        Resize((cfg.params['imagex'], cfg.params['imagey'], cfg.params['imagez']))  # Resize to target dimensions
    ])
    # Same transforms for validation (no augmentation needed)
    val_transforms = Compose([
        ScaleIntensity(), 
        EnsureChannelFirst(), 
        Resize((cfg.params['imagex'], cfg.params['imagey'], cfg.params['imagez']))
    ])
    
    # ==================== Dataset and DataLoader Setup ====================
    # Create PyTorch datasets
    train_ds = ImageDataset(image_files=image_train, labels=label_train, transform=train_transforms)
    val_ds = ImageDataset(image_files=image_val, labels=label_val, transform=val_transforms)
    
    # Create data loaders for batch processing
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.params['batch_size'],     # Batch size from config
        shuffle=True,                            # Shuffle training data each epoch
        generator=g,                             # Use seeded generator for reproducibility
        pin_memory=torch.cuda.is_available()     # Speed up GPU transfer if CUDA available
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.params['batch_size'], 
        shuffle=False,                           # Don't shuffle validation data
        generator=g, 
        pin_memory=torch.cuda.is_available()
    )
    
    # Calculate number of batches per epoch (for logging)
    train_epoch_len = len(train_ds) // train_loader.batch_size
    val_epoch_len = len(val_ds) // val_loader.batch_size

    # ==================== Model Initialization ====================
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize SFCN model and move to device
    model = get_model('sfcn', spatial_dims=3, in_channels=1, out_channels=2).to(device)
    
    # Print model architecture summary
    summary(model, (cfg.params['batch_size'], 1, cfg.params['imagex'], cfg.params['imagey'], cfg.params['imagez']))
    
    # ==================== Loss Function and Optimizer ====================
    loss_function = torch.nn.CrossEntropyLoss()  # Standard loss for classification
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.params['lr'],      # Learning rate from config
        weight_decay=0.001        # L2 regularization
    )

    # ==================== Training Loop Setup ====================
    # Initialize variables for early stopping and metric tracking
    patience_counter = 0                    # Counter for epochs without improvement
    best_val_loss = float('inf')            # Best validation loss seen so far
    train_epoch_loss_values = []            # List to store training losses
    val_epoch_loss_values = []              # List to store validation losses

    # ==================== Main Training Loop ====================
    for epoch in range(cfg.params['epochs']):
        print("-"*10)
        print(f"epoch {epoch+1}/{cfg.params['epochs']}")
        print(f"Patience: {patience_counter}/{cfg.params['patience']}")

        # ==================== Training Phase ====================
        train_epoch_loss = 0     # Accumulator for training loss
        train_correct = 0        # Counter for correct predictions
        train_total = 0          # Counter for total samples
        model.train()            # Set model to training mode (enables dropout, batch norm updates)
        
        # Iterate through training batches
        for step, batch_data in enumerate(train_loader, 1):
            # Move data to device
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_function(outputs, labels.long())
            
            # Backward pass (compute gradients)
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            
            # Accumulate loss for averaging
            train_epoch_loss += loss.item()
            
            # Calculate batch accuracy
            _, predicted = torch.max(outputs, 1)  # Get predicted class (0 or 1)
            train_correct += (predicted == labels).sum().item()  # Count correct predictions
            train_total += labels.size(0)  # Count total samples in batch
            
            # Print progress every batch
            print(f"{step}/{train_epoch_len}, train_loss: {loss.item():.4f}")

        # Calculate epoch-level training metrics
        train_accuracy = train_correct / train_total  # Average accuracy across all batches
        train_epoch_loss /= train_epoch_len           # Average loss across all batches
        train_epoch_loss_values.append(train_epoch_loss)  # Store for plotting
        print(f"Training Accuracy: {train_accuracy:.4f}, Avg Train Loss: {train_epoch_loss:.4f}")

        # ==================== Validation Phase ====================
        val_epoch_loss = 0       # Accumulator for validation loss
        val_correct = 0          # Counter for correct predictions
        val_total = 0            # Counter for total samples
        model.eval()             # Set model to evaluation mode (disables dropout, batch norm in eval mode)
        
        # Disable gradient computation for validation (saves memory and computation)
        with torch.no_grad():
            # Iterate through validation batches
            for step, val_data in enumerate(val_loader, 1):
                # Move data to device
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                
                # Forward pass (no backward pass in validation)
                val_outputs = model(val_images)
                
                # Compute loss
                val_loss = loss_function(val_outputs, val_labels.long())
                val_epoch_loss += val_loss.item()
                
                # Calculate batch accuracy
                _, predicted = torch.max(val_outputs, 1)
                val_correct += (predicted == val_labels).sum().item()
                val_total += val_labels.size(0)
                
                # Print progress
                print(f"{step}/{val_epoch_len}, val_loss: {val_loss.item():.4f}")

        # Calculate epoch-level validation metrics
        val_accuracy = val_correct / val_total
        val_epoch_loss /= step
        val_epoch_loss_values.append(val_epoch_loss)
        print(f"Validation Accuracy: {val_accuracy:.4f}, Avg Val Loss: {val_epoch_loss:.4f}")

        # ==================== Model Checkpointing ====================
        # Save model if validation loss improved
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss      # Update best validation loss
            best_metric_epoch = epoch + 1        # Record epoch number
            
            # Save model state dictionary (weights only)
            torch.save(model.state_dict(), model_weight_path)
            
            # Save complete checkpoint (includes optimizer state for potential resume)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(outdir, f'Best_model_checkpoint_{exp_name}.pth'))
            
            # Save complete model object (architecture + weights)
            torch.save(model, os.path.join(outdir, f'Best_model_checkpoint_{exp_name}.model'))
            
            # Reset patience counter (model improved)
            patience_counter = 0
        else:
            # No improvement - increment patience counter
            patience_counter += 1
            
            # Write text file with best epoch info
            txt_path = os.path.join(outdir, f'Best_model_checkpoint_{exp_name}.txt')
            with open(txt_path, 'w') as f:
                f.write(f"Best model saved at epoch: {best_metric_epoch}\n")

        # ==================== Early Stopping Check ====================
        # Stop training if no improvement for 'patience' epochs
        if patience_counter >= cfg.params['patience']:
            print(f"Early stopping at epoch {epoch+1}, best val loss: {best_val_loss:.4f} at epoch {best_metric_epoch}")
            break  # Exit training loop
            
        # ==================== Save Metrics ====================
        # Save accuracy to CSV (append mode)
        accuracy_data = {
            'epoch': epoch+1, 
            'train_accuracy': train_accuracy, 
            'val_accuracy': val_accuracy
        }
        accuracy_df = pd.DataFrame([accuracy_data])
        accuracy_csv_path = os.path.join(outdir, f'accuracy_log_{exp_name}.csv')
        # Append to CSV, write header only if file doesn't exist
        accuracy_df.to_csv(accuracy_csv_path, mode='a', header=not os.path.exists(accuracy_csv_path), index=False)
        
        # ==================== Plot Loss Curves ====================
        # Create and save loss curve plot
        plt.figure(figsize=(18,12))
        plt.subplot(2,1,1)
        plt.plot(range(1, len(train_epoch_loss_values)+1), train_epoch_loss_values, '-o', color='red', label="Training Epoch Loss")
        plt.plot(range(1, len(val_epoch_loss_values)+1), val_epoch_loss_values, '-o', color='blue', label="Validation Epoch Loss")
        plt.title("Epoch Loss vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Epoch Loss")
        plt.legend()
        plt.grid(True)
        png_path = os.path.join(outdir, f'validation_metrics_{exp_name}.png')
        plt.savefig(png_path)
        plt.close()  # Close figure to free memory


def main():
    """
    Main function to run 20-fold cross-validation.
    
    Trains the model on all 20 folds sequentially using a fixed seed
    for reproducibility across folds.
    """
    seed = 7778  # Fixed seed for all folds
    
    # Train each fold from 1 to 20
    for fold_number in range(1, 21):
        print(f"\n\n===== Training fold {fold_number} =====\n\n")
        train_fold(fold_number, seed)


# ==================== Script Entry Point ====================
if __name__ == "__main__":
    main()
