#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Saliency Map Generation Pipeline for Explainable Autism Classification
======================================================================
This script generates and processes SmoothGrad saliency maps to explain
model predictions on structural MRI scans. The complete pipeline includes:

1. Generate individual saliency maps using SmoothGrad (150 samples)
2. Copy spatial metadata from reference atlas
3. Register saliency maps to MNI standard space using NiftyReg
4. Aggregate all saliency maps by summation
5. Apply percentile thresholding to remove noise
6. Mask with brain atlas to keep only brain tissue

Requirements:
- Trained SFCN model (.model file)
- CSV file with participant IDs and MRI paths
- Reference atlas in MNI space (.nii.gz)
- Structural images for registration
- NiftyReg installed (for reg_f3d and reg_resample commands)

Usage:
    python generation_of_saliency_maps.py
    
    Note: Update all paths in the "PATH DEFINITIONS" section before running

Output Structure:
    BASE_DIR/
    ├── first/                  # Raw saliency maps
    ├── correct/                # Spatially corrected maps
    ├── DEFORMATIONS/           # Registered maps + deformation fields
    ├── TEST_AV_SMAP.nii.gz            # Aggregated (summed) map
    ├── nn_TEST_AV_SMAP.nii.gz         # Thresholded map
    └── masked_nn_TEST_AV_SMAP.nii.gz  # Final masked map
"""

import os
import torch
import random
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import subprocess
import glob
from monai.visualize import SmoothGrad  # MONAI's SmoothGrad implementation
from SFCN_Class import SFCN


# ==========================================================
# 1. SALIENCY MAP COMPUTATION FUNCTIONS
# ==========================================================

def compute_saliency_map(model, input_tensor):
    """
    Compute SmoothGrad saliency map for a single subject.
    
    SmoothGrad reduces noise in gradient-based explanations by averaging
    gradients over multiple noisy versions of the input image. This produces
    smoother, more interpretable visualizations of important brain regions.
    
    Args:
        model (torch.nn.Module): Trained SFCN model in evaluation mode
        input_tensor (torch.Tensor): Input MRI volume with shape (1, 1, D, H, W)
    
    Returns:
        numpy.ndarray: 3D saliency map showing importance of each voxel
                      Shape: (D, H, W)
    
    Note:
        - Uses 150 noisy samples for robust gradient averaging
        - Runs on CPU to avoid CUDA memory issues with large volumes
        - Requires gradients to be enabled on input_tensor
    """
    # Initialize SmoothGrad with 150 noisy samples
    # More samples = smoother map but slower computation
    guided_backprop = SmoothGrad(model, n_samples=150)
    
    # Use CPU for saliency computation (more stable for large 3D volumes)
    device = torch.device("cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Enable gradient computation for input (required for saliency)
    input_tensor.requires_grad = True
    
    # Compute saliency map using SmoothGrad
    # This averages gradients over 150 noisy versions of the input
    saliency_map = guided_backprop(input_tensor)
    
    # Convert to numpy array and remove batch/channel dimensions
    # squeeze() removes dimensions of size 1: (1,1,D,H,W) → (D,H,W)
    saliency_map = saliency_map.squeeze().detach().cpu().numpy()
    
    return saliency_map


def save_saliency_map_as_nifti(saliency_map, affine, output_path):
    """
    Save a saliency map as a NIfTI image file.
    
    NIfTI format preserves spatial information (voxel size, orientation)
    needed for visualization and further processing.
    
    Args:
        saliency_map (numpy.ndarray): 3D saliency map array
        affine (numpy.ndarray): 4x4 affine transformation matrix 
                               (defines voxel-to-world coordinate mapping)
        output_path (str): Path where NIfTI file will be saved
    """
    # Create NIfTI image object with saliency data and spatial info
    nifti_img = nib.Nifti1Image(saliency_map, affine)
    
    # Save to disk
    nib.save(nifti_img, output_path)


# ==========================================================
# 2. PATH DEFINITIONS
# ==========================================================
# IMPORTANT: Update these paths to match your directory structure

BASE_DIR = "/home/miplab/Desktop/CLEANING/SALIENCY_MAPS"  # Main output directory - CHANGE THIS
MODEL_PATH = f"{BASE_DIR}/FOLD_MAMI/Best_model_checkpoint_fold_5.model"  # Trained model - CHANGE THIS
CSV_PATH = f"{BASE_DIR}/FOLD_MAMI/CC_test.csv"  # Subject list CSV - CHANGE THIS

# Output directories for different pipeline stages
SALIENCY_RAW_DIR = f"{BASE_DIR}/first"              # Stage 1: Raw saliency maps
SALIENCY_CORRECTED_DIR = f"{BASE_DIR}/correct"      # Stage 2: Spatially corrected maps
DEFORMATIONS_DIR = f"{BASE_DIR}/DEFORMATIONS"       # Stage 3: Registered maps

# Input paths for registration
STRUCTURAL_IMAGES_DIR = "/home/miplab/Documents/AUTISM_STUDY/NIFTY"  # Original structural MRIs - CHANGE THIS
ATLAS_PATH = f"{BASE_DIR}/niphd_PERMUTED_CROPPED.nii.gz"  # Reference atlas (MNI space) - CHANGE THIS

# Output paths for aggregated maps
SUM_OUTPUT_PATH = f"{BASE_DIR}/TEST_AV_SMAP.nii.gz"        # Summed saliency map
CLEANED_SMAP_PATH = f"{BASE_DIR}/nn_TEST_AV_SMAP.nii.gz"   # Thresholded map
MASKED_SMAP_PATH = f"{BASE_DIR}/masked_nn_TEST_AV_SMAP.nii.gz"  # Final masked map


# ==========================================================
# 3. MAIN EXECUTION SECTION
# ==========================================================

if __name__ == '__main__':

    # ==================== Load Trained Model ====================
    print("Loading trained model...")
    device = torch.device("cpu")  # Use CPU for stability with large 3D volumes
    
    # Load complete model object (architecture + weights)
    model = torch.load(MODEL_PATH, map_location=device)
    model.to(device)
    model.eval()  # Set to evaluation mode (disables dropout, batch norm updates)
    print("Model loaded successfully!")

    # ==================== Load Subject List ====================
    print(f"\nLoading subject list from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Found {len(df)} subjects to process")
    
    # Expected CSV columns:
    # - 'Link': Absolute path to preprocessed MRI (.nii.gz)
    # - 'participant_id': Subject identifier (e.g., 'sub-0001')

    # ==================== Generate Individual Saliency Maps ====================
    print("\n" + "="*60)
    print("STAGE 1: Generating individual saliency maps")
    print("="*60)
    
    # Iterate through each subject in the dataframe
    for index, row in df.iterrows():
        nifti_path = row['Link']           # Path to subject's MRI
        subject_name = row['participant_id']  # Subject ID
        print(f"\nProcessing: {subject_name}")

        # Load NIfTI image
        nifti_img = nib.load(nifti_path)
        input_image = nifti_img.get_fdata()  # Get 3D array of voxel intensities
        affine = nifti_img.affine            # Get spatial transformation matrix
        
        # Convert to PyTorch tensor and add batch + channel dimensions
        # Shape: (D,H,W) → (1,1,D,H,W)
        input_tensor = torch.from_numpy(input_image).float()
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

        # Compute saliency map for this subject
        saliency_map = compute_saliency_map(model, input_tensor)

        # Save saliency map as NIfTI
        output_path = f"{SALIENCY_RAW_DIR}/Saliency_Map_{subject_name}.nii.gz"
        save_saliency_map_as_nifti(saliency_map, affine, output_path)
        print(f"Saved: {output_path}")

        # Clean up memory (important when processing many subjects)
        del saliency_map, input_tensor, nifti_img
        torch.cuda.empty_cache()  # Clear CUDA cache if GPU was used

    print("\n✓ All individual saliency maps generated!")

    # ==========================================================
    # 4. COPY ATLAS METADATA INTO SALIENCY MAPS
    # ==========================================================
    print("\n" + "="*60)
    print("STAGE 2: Correcting spatial information")
    print("="*60)
    
    # Create output directory for corrected maps
    os.makedirs(SALIENCY_CORRECTED_DIR, exist_ok=True)
    
    # Load atlas image (reference for spatial information)
    atlas_image = sitk.ReadImage(ATLAS_PATH)
    print(f"Using atlas: {ATLAS_PATH}")

    # Copy spatial information from atlas to each saliency map
    # This ensures saliency maps have correct:
    # - Voxel spacing (physical size of voxels)
    # - Origin (position in world coordinates)
    # - Direction (orientation matrix)
    for filename in os.listdir(SALIENCY_RAW_DIR):
        if filename.endswith(".nii.gz"):
            # Read saliency map
            saliency_map_path = os.path.join(SALIENCY_RAW_DIR, filename)
            saliency_map = sitk.ReadImage(saliency_map_path)
            
            # Copy spatial metadata from atlas
            saliency_map.CopyInformation(atlas_image)
            
            # Save corrected map
            output_path = os.path.join(
                SALIENCY_CORRECTED_DIR,
                filename.replace(".nii.gz", "_corrected.nii.gz")
            )
            sitk.WriteImage(saliency_map, output_path)
            print(f"Corrected: {filename}")

    print("\n✓ All saliency maps spatially corrected!")

    # ==========================================================
    # 5. WARP SALIENCY MAPS TO ATLAS SPACE USING NIFTYREG
    # ==========================================================
    print("\n" + "="*60)
    print("STAGE 3: Registering saliency maps to MNI space")
    print("="*60)
    print("Note: This step requires NiftyReg to be installed")
    
    # Get list of subject IDs from dataframe
    participant_ids = df['participant_id'].tolist()
    print(f"Registering {len(participant_ids)} subjects...")

    # Process each corrected saliency map
    for subject_file in os.listdir(SALIENCY_CORRECTED_DIR):
        # Only process corrected saliency map files
        if subject_file.endswith("_corrected.nii.gz") and subject_file.startswith("Saliency_Map_sub-"):
            # Extract participant ID from filename
            # e.g., "Saliency_Map_sub-0001_corrected.nii.gz" → "sub-0001"
            participant_id = "sub-" + subject_file.split("-")[1].split("_corrected")[0]
            print(f"\nProcessing: {participant_id}")

            # Skip if participant not in our list
            if participant_id not in participant_ids:
                print(f"  Skipping {participant_id} (not in participant list)")
                continue

            # Define file paths for registration
            structural_image = os.path.join(STRUCTURAL_IMAGES_DIR, f"{participant_id}.nii.gz")  # Original MRI
            saliency_map = os.path.join(SALIENCY_CORRECTED_DIR, subject_file)  # Saliency map to warp
            deformation_field = os.path.join(DEFORMATIONS_DIR, f"Saliency_Map_sub-{participant_id}_deformation_field.nii.gz")  # Transformation
            registered_image = os.path.join(DEFORMATIONS_DIR, f"Saliency_Map_sub-{participant_id}_registered.nii.gz")  # Registered structural
            resampled_image = os.path.join(DEFORMATIONS_DIR, f"Saliency_Map_sub-{participant_id}_resampled.nii.gz")  # Warped saliency map

            # Check if required files exist
            if not os.path.isfile(structural_image):
                print(f"  ⚠ Structural image not found for {participant_id}")
                continue
            if not os.path.isfile(saliency_map):
                print(f"  ⚠ Saliency map not found for {participant_id}")
                continue

            # ==================== Step 5a: Register structural image to atlas ====================
            # This creates a deformation field describing how to warp subject space → atlas space
            print(f"  Running registration (reg_f3d)...")
            reg_f3d_cmd = f"reg_f3d -ref {ATLAS_PATH} -flo {structural_image} -cpp {deformation_field} -res {registered_image}"
            subprocess.run(reg_f3d_cmd, shell=True, check=True)

            # ==================== Step 5b: Apply deformation to saliency map ====================
            # Warp the saliency map using the deformation field from step 5a
            print(f"  Resampling saliency map (reg_resample)...")
            reg_resample_cmd = f"reg_resample -ref {ATLAS_PATH} -flo {saliency_map} -trans {deformation_field} -res {resampled_image}"
            subprocess.run(reg_resample_cmd, shell=True, check=True)

            print(f"  ✓ Completed: {participant_id}")

    print("\n✓ All saliency maps registered to MNI space!")

    # ==========================================================
    # 6. SUM ALL WARPED SALIENCY MAPS
    # ==========================================================
    print("\n" + "="*60)
    print("STAGE 4: Aggregating saliency maps")
    print("="*60)
    
    # Find all resampled (warped) saliency maps
    saliency_map_files = glob.glob(DEFORMATIONS_DIR + '/Saliency_Map_sub*.nii.gz')
    print(f"Found {len(saliency_map_files)} registered saliency map files")
    
    # Create dictionary mapping participant IDs to their saliency map files
    participant_to_saliency_map = {}
    for _, row in df.iterrows():
        pid = row['participant_id']
        # Look for the resampled file for this participant
        for file in saliency_map_files:
            if f'Saliency_Map_sub-{pid}_resampled.nii.gz' in file:
                participant_to_saliency_map[pid] = file
                print(f"  ✓ Found file for {pid}")
                break

    print(f"\nAggregating {len(participant_to_saliency_map)} saliency maps...")

    # Load first file to get dimensions
    first_file = list(participant_to_saliency_map.values())[0]
    sum_saliency_map = np.zeros_like(nib.load(first_file).get_fdata())

    # Sum all saliency maps (element-wise addition)
    for pid, file in participant_to_saliency_map.items():
        print(f"  Adding: {pid}")
        saliency_map = nib.load(file).get_fdata()
        sum_saliency_map += saliency_map  # Accumulate saliency values

    # Save aggregated (summed) saliency map
    nib.save(nib.Nifti1Image(sum_saliency_map, affine=np.eye(4)), SUM_OUTPUT_PATH)
    print(f"\n✓ Aggregated map saved: {SUM_OUTPUT_PATH}")

    # ==========================================================
    # 7. QUANTILE FILTER + MASKING
    # ==========================================================
    print("\n" + "="*60)
    print("STAGE 5: Thresholding and masking")
    print("="*60)
    
    # ==================== Step 7a: Apply percentile threshold ====================
    # Remove low-intensity noise by setting values below 25th percentile to zero
    print("Applying 25th percentile threshold...")
    avg_smap = sitk.ReadImage(SUM_OUTPUT_PATH)
    avg_np = sitk.GetArrayFromImage(avg_smap)
    
    # Calculate 25th percentile (lower quartile)
    lower_quartile = np.percentile(avg_np, 25)
    print(f"  Lower quartile value: {lower_quartile:.4f}")
    
    # Threshold: Keep only values >= lower quartile
    cleaned_smap = np.where(avg_np < lower_quartile, 0, avg_np)
    
    # Save thresholded map
    cleaned_smap_img = sitk.GetImageFromArray(cleaned_smap)
    cleaned_smap_img.CopyInformation(avg_smap)  # Preserve spatial info
    sitk.WriteImage(cleaned_smap_img, CLEANED_SMAP_PATH)
    print(f"  ✓ Thresholded map saved: {CLEANED_SMAP_PATH}")

    # ==================== Step 7b: Apply brain mask ====================
    # Keep only voxels that are within brain tissue (atlas > 0)
    print("\nApplying brain mask...")
    saliency_map_nii = nib.load(CLEANED_SMAP_PATH)
    segmentation_nii = nib.load(ATLAS_PATH)
    sm_data = saliency_map_nii.get_fdata()
    seg_data = segmentation_nii.get_fdata()

    # Element-wise multiplication: Keep saliency only where atlas has brain tissue
    masked_smap = sm_data * (seg_data > 0)
    
    # Save final masked saliency map
    nib.save(
        nib.Nifti1Image(masked_smap, affine=saliency_map_nii.affine, header=saliency_map_nii.header),
        MASKED_SMAP_PATH
    )

    print(f"  ✓ Final masked map saved: {MASKED_SMAP_PATH}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nFinal saliency map statistics:")
    print(f"  Min value: {masked_smap.min():.4f}")
    print(f"  Max value: {masked_smap.max():.4f}")
    print(f"  Mean value: {masked_smap.mean():.4f}")
    print(f"  Non-zero voxels: {np.count_nonzero(masked_smap)}")
    print(f"\nOutput location: {MASKED_SMAP_PATH}")
    
    # Exit program
    exit()
