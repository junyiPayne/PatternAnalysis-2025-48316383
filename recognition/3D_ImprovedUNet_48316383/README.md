# 3D Improved UNet for Prostate Segmentation

Student ID: 48316383

## Overview
Implementation of 3D Improved UNet for semantic segmentation of prostate MRI scans. Target: Dice coefficient ≥ 0.7 for all labels on test set.

## Dataset
- Prostate 3D MRI dataset (NIfTI format)
- 5 classes: background + 4 anatomical structures
- Data augmentation: rotations, flips, elastic deformations

## Model Architecture
- 3D UNet with skip connections
- Encoder: 3 downsampling stages (32→64→128 channels)
- Bottleneck: 256 channels
- Decoder: 3 upsampling stages with concatenation
- Output: 5-channel segmentation map

## Training
```bash
python train.py
```
- Loss: Dice Loss
- Optimizer: Adam (lr=1e-4)
- Scheduler: ReduceLROnPlateau
- Batch size: 1 (due to 3D memory constraints)

## Prediction
```bash
python predict.py
```

## Requirements
- PyTorch >= 1.10
- nibabel
- numpy
- tqdm

## References
- [1] 3D Improved UNet
- [7] CAN3D
- [8] 3D UNet (Çiçek et al., 2016)