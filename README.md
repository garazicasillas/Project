# Project title: Interpretable convolutional neural network for autism diagnosis support in children using structural magnetic resonance imaging datasets


A 3D deep learning pipeline for binary classification of Autism Spectrum Disorder (ASD) from structural brain MRI scans with integrated explainability through SmoothGrad saliency maps.
Study title: Interpretable convolutional neural network for autism diagnosis support in children using structural magnetic resonance imaging datasets

## 📋 Overview

This repository implements a **Simple Fully Convolutional Network (SFCN)** with progressive dropout for classifying autism from 3D structural MRI scans. The model is trained and evaluated using **20-fold cross-validation** on the ABIDE (Autism Brain Imaging Data Exchange) dataset, and includes explainability features through **SmoothGrad saliency maps** to identify brain regions contributing to classification decisions.

### Key Features

- ✅ **SFCN Architecture**: Lightweight 3D CNN optimized for volumetric medical imaging
- ✅ **Progressive Dropout**: Increasing dropout rates (0.05→0.25) across layers for regularization
- ✅ **20-Fold Cross-Validation**: Robust evaluation with independent fold training
- ✅ **Explainability**: SmoothGrad-based saliency maps for interpretable predictions
- ✅ **Registration Pipeline**: Automated MNI space normalization using NiftyReg
- ✅ **Reproducibility**: Fixed seed control for deterministic training

---

## 🏗️ Architecture

### SFCN Model Structure

```
Input (1×161×201×165)
    ↓
Block 1: Conv3D(16) + BatchNorm + MaxPool + ReLU
    ↓
Block 2: Conv3D(32) + BatchNorm + MaxPool + ReLU + Dropout(0.05)
    ↓
Block 3: Conv3D(64) + BatchNorm + MaxPool + ReLU + Dropout(0.1)
    ↓
Block 4: Conv3D(128) + BatchNorm + MaxPool + ReLU + Dropout(0.15)
    ↓
Block 5: Conv3D(64, 1×1×1) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Block 6: AvgPool + Dropout(0.25) + Flatten + Linear(2)
    ↓
Output (Autism, Control)
```

**Key Design Choices:**
- Progressive dropout prevents overfitting while maintaining early-layer learning
- 1×1×1 convolution in Block 5 reduces channels before classification
- Global average pooling reduces spatial dimensions
- LazyLinear automatically infers input size

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works)
- ~20GB free disk space
- NiftyReg (for saliency map registration)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/autism-mri-classification.git
cd autism-mri-classification
```

### Step 2: Install Dependencies

```bash
pip install torch torchvision
pip install monai nibabel SimpleITK
pip install pandas numpy matplotlib seaborn
pip install scikit-learn tqdm torchinfo
```

Or use the provided requirements file:
```bash
pip install -r requirements.txt
```

### Step 3: Install NiftyReg (for saliency maps)

**Ubuntu/Debian:**
```bash
sudo apt-get install niftyreg
```

**macOS:**
```bash
brew install niftyreg
```

**Verify installation:**
```bash
reg_f3d --version
reg_resample --version
```

---

## 📊 Dataset Preparation

### ABIDE Dataset

This project uses the [ABIDE (Autism Brain Imaging Data Exchange)](http://fcon_1000.projects.nitrc.org/indi/abide/) dataset, a publicly available collection of structural and functional MRI data.

### Data Organization

Organize your data as follows:

```
data/
├── 20FOLD_TIPICAL_CV/
│   ├── train_fold_1.csv
│   ├── val_fold_1.csv
│   ├── train_fold_2.csv
│   ├── val_fold_2.csv
│   └── ... (up to fold 20)
└── structural_images/
    ├── sub-0001.nii.gz
    ├── sub-0002.nii.gz
    └── ...
```

### CSV Format

Each CSV file should contain:

| participant_id | Link | DX_GROUP |
|---------------|------|----------|
| sub-0001 | /absolute/path/to/sub-0001.nii.gz | 0 |
| sub-0002 | /absolute/path/to/sub-0002.nii.gz | 1 |

- `participant_id`: Unique subject identifier
- `Link`: **Absolute path** to preprocessed NIfTI file
- `DX_GROUP`: Label (0 = Control, 1 = Autism)

### Preprocessing

Recommended preprocessing steps:

1. **Skull stripping**: Remove non-brain tissue (using FSL BET or FreeSurfer)
2. **Bias field correction**: Correct intensity inhomogeneities (using ANTs N4)
3. **Registration**: Align to MNI152 template (using ANTs or FSL FLIRT)
4. **Intensity normalization**: Scale intensities to [0, 1]
5. **Cropping/Resampling**: Resize to 161×201×165 voxels

---

## 🎯 Usage

### Training with 20-Fold Cross-Validation

#### Step 1: Configure Paths

Edit the following paths in `my_training_model.py`:

```python
# Line 67: Output directory
outdir = f"/path/to/your/output/{fold_number}_seed_{seed}"

# Line 80: Data directory containing CSV files
data_dir = '/path/to/your/20FOLD_TIPICAL_CV'
```

#### Step 2: Adjust Hyperparameters (Optional)

Edit `config_file.py` to modify training settings:

```python
params = {
    'batch_size': 2,      # Reduce to 1 if GPU memory issues
    'lr': 0.00001,        # Learning rate
    'epochs': 130,        # Maximum epochs (early stopping may stop earlier)
    'patience': 30,       # Early stopping patience
    'imagex': 161,        # Image dimensions
    'imagey': 201,
    'imagez': 165,
}
```

#### Step 3: Run Training

```bash
python my_training_model.py
```

This will train all 20 folds sequentially with seed 7778.

### Training Outputs

For each fold, the script generates:

```
{fold_number}_seed_7778/
├── Best_model_weights_fold_{N}.pth         # Model weights (state dict)
├── Best_model_checkpoint_fold_{N}.pth      # Full checkpoint (weights + optimizer)
├── Best_model_checkpoint_fold_{N}.model    # Complete model object
├── Best_model_checkpoint_fold_{N}.txt      # Best epoch info
├── accuracy_log_fold_{N}.csv               # Per-epoch accuracy
└── validation_metrics_fold_{N}.png         # Loss curves
```

---

### Generating Saliency Maps

#### Step 1: Configure Paths

Edit paths in `generation_of_saliency_maps.py`:

```python
# Lines 145-159: Update all paths
BASE_DIR = "/path/to/your/saliency_output"
MODEL_PATH = f"{BASE_DIR}/FOLD_MAMI/Best_model_checkpoint_fold_5.model"
CSV_PATH = f"{BASE_DIR}/FOLD_MAMI/CC_test.csv"
STRUCTURAL_IMAGES_DIR = "/path/to/structural_images"
ATLAS_PATH = f"{BASE_DIR}/niphd_PERMUTED_CROPPED.nii.gz"
```

#### Step 2: Prepare Files

Ensure you have:
- ✅ Trained model (`.model` file)
- ✅ CSV file with test subjects
- ✅ Reference atlas in MNI space
- ✅ Original structural images for registration

#### Step 3: Run Pipeline

```bash
python generation_of_saliency_maps.py
```

### Pipeline Stages

The script executes 5 stages automatically:

1. **Generate Individual Maps**: Compute SmoothGrad for each subject (150 samples)
2. **Spatial Correction**: Copy atlas metadata to saliency maps
3. **Registration to MNI**: Warp maps to standard space using NiftyReg
4. **Aggregation**: Sum all registered maps
5. **Thresholding & Masking**: Apply 25th percentile threshold and brain mask

### Saliency Map Outputs

```
BASE_DIR/
├── first/                              # Raw saliency maps
│   └── Saliency_Map_sub-*.nii.gz
├── correct/                            # Spatially corrected
│   └── Saliency_Map_sub-*_corrected.nii.gz
├── DEFORMATIONS/                       # Registered maps
│   ├── Saliency_Map_sub-*_resampled.nii.gz
│   ├── Saliency_Map_sub-*_deformation_field.nii.gz
│   └── Saliency_Map_sub-*_registered.nii.gz
├── TEST_AV_SMAP.nii.gz                # Aggregated map (sum)
├── nn_TEST_AV_SMAP.nii.gz             # Thresholded map
└── masked_nn_TEST_AV_SMAP.nii.gz      # Final result ⭐
```

---

## 📈 Results Interpretation

### Training Metrics

Monitor training progress through:

1. **Console output**: Real-time loss and accuracy per batch
2. **CSV logs**: `accuracy_log_fold_{N}.csv` contains per-epoch metrics
3. **Loss curves**: `validation_metrics_fold_{N}.png` visualizes training/validation loss


### Saliency Map Interpretation

- **Bright regions**: High importance for classification
- **Dark regions**: Low importance
- **Common regions**: Temporal cortex, cerebellum, frontal lobes, amygdala

Use neuroimaging software to visualize:

```bash
# FSLeyes
fsleyes masked_nn_TEST_AV_SMAP.nii.gz -cm hot -dr 0 1000

# ITK-SNAP
itksnap -g MNI152_T1_1mm.nii.gz -o masked_nn_TEST_AV_SMAP.nii.gz
```

---

## 📁 Project Structure

```
autism-mri-classification/
│
├── config_file.py                      # Hyperparameters and configuration
├── SFCN_Class.py                      # SFCN model architecture
├── utils.py                            # Model loading utilities
├── my_training_model.py               # 20-fold CV training script
├── generation_of_saliency_maps.py     # Explainability pipeline
│
├── data/                               # Data directory (user-created)
│   ├── 20FOLD_TIPICAL_CV/
│   │   ├── train_fold_*.csv
│   │   └── val_fold_*.csv
│   └── structural_images/
│       └── sub-*.nii.gz
│
└── results/                            # Training outputs (generated)
    └── {fold}_seed_{seed}/
        ├── *.pth
        ├── *.csv
        └── *.png
```

---

## ⚙️ Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 2 | Samples per batch (reduce if OOM) |
| `lr` | 0.00001 | Learning rate |
| `epochs` | 130 | Maximum epochs |
| `patience` | 30 | Early stopping patience |
| `imagex/y/z` | 161/201/165 | Target image dimensions |

### Model Parameters

| Layer | Channels | Dropout | Purpose |
|-------|----------|---------|---------|
| Conv1 | 16 | - | Low-level features |
| Conv2 | 32 | 0.05 | Mid-level features |
| Conv3 | 64 | 0.1 | High-level features |
| Conv4 | 128 | 0.15 | Abstract features |
| Conv5 | 64 | 0.2 | Channel reduction |
| Final | 2 | 0.25 | Classification |

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# In config_file.py, change:
'batch_size': 1  # Reduce from 2
```

**2. NiftyReg Not Found**
```bash
# Check installation
which reg_f3d

# Add to PATH if needed
export PATH=$PATH:/usr/local/bin
```

**3. Import Errors**
```python
# Ensure all files are in the same directory:
SFCN_Class.py
utils.py
config_file.py
```

**4. CSV Loading Errors**
```python
# Verify CSV format
import pandas as pd
df = pd.read_csv('train_fold_1.csv')
print(df.columns.tolist())  # Should show: ['participant_id', 'Link', 'DX_GROUP']
print(df['Link'].iloc[0])   # Should be absolute path
```

---

## 📚 References

### SFCN Architecture

```bibtex
@article{peng2021accurate,
  title={Accurate brain age prediction with lightweight deep neural networks},
  author={Peng, Han and Gong, Weikang and Beckmann, Christian F and Vedaldi, Andrea and Smith, Stephen M},
  journal={Medical image analysis},
  volume={68},
  pages={101871},
  year={2021}
}
```

### ABIDE Dataset

```bibtex
@article{di2014autism,
  title={The autism brain imaging data exchange: towards large-scale evaluation of the intrinsic brain architecture in autism},
  author={Di Martino, Adriana and Yan, Chao-Gan and Li, Qingyang and others},
  journal={Molecular psychiatry},
  volume={19},
  number={6},
  pages={659--667},
  year={2014}
}
```

### SmoothGrad

```bibtex
@article{smilkov2017smoothgrad,
  title={SmoothGrad: removing noise by adding noise},
  author={Smilkov, Daniel and Thorat, Nikhil and Kim, Been and Vi{\'e}gas, Fernanda and Wattenberg, Martin},
  journal={arXiv preprint arXiv:1706.03825},
  year={2017}
}
```

---

## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 [Garazi Casillas Martinez]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

- **ABIDE Consortium**: For providing open-access autism neuroimaging data
- **MONAI**: For medical imaging deep learning framework  
- **PyTorch**: For deep learning infrastructure
- **NiftyReg**: For medical image registration tools

---


