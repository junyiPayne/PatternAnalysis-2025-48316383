# Quick Reference Guide

## Essential Commands

### Environment Setup
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Pipeline Execution
```powershell
# 1. Validate dataset
python check_data.py

# 2. Train model
python train.py

# 3. Run inference
python predict.py

# 4. Generate visualizations
python visualize.py
```

## File Structure

```
📁 Project Root
├── config.py              # ⚙️  Configuration (EDIT THIS FIRST)
├── modules.py             # 🏗️  Model architecture
├── dataset.py             # 📦 Data loading
├── train.py               # 🎯 Training script (RUN THIS)
├── predict.py             # 🔮 Inference script
├── visualize.py           # 🎨 Visualization script
├── utils.py               # 🛠️  Utilities
├── training_visualizer.py # 📊 Metrics tracking
├── check_data.py          # ✅ Data validation
├── requirements.txt       # 📋 Dependencies
├── README.md              # 📖 Full documentation
└── INSTALLATION.md        # 💿 Installation guide
```

## Key Configuration Parameters

### In `config.py`:

```python
# MODEL
'base_filters': 24,          # 16 (small) | 24 (medium) | 32 (large)
'num_classes': 6,            # Don't change for HipMRI
'target_size': (128,128,64), # Reduce if OOM

# TRAINING
'num_epochs': 20,            # Training duration
'train_batch_size': 2,       # Reduce to 1 if OOM
'learning_rate': 1e-3,       # 1e-4 | 5e-4 | 1e-3 | 2e-3
'use_amp': True,             # Mixed precision (KEEP TRUE)

# PATHS
'data_root': r"C:\path\to\data",  # UPDATE THIS!
```

## Common Tasks

### Change Training Duration
```python
# In config.py
'num_epochs': 50,  # Longer training
```

### Reduce Memory Usage
```python
# In config.py
'train_batch_size': 1,       # Instead of 2
'base_filters': 16,          # Instead of 24
'target_size': (96,96,48),   # Instead of (128,128,64)
```

### Disable Visualization
```python
# In config.py
'enable_visualization': False,
```

### Change Learning Rate
```python
# In config.py
'learning_rate': 5e-4,  # Try different values
```

## Output Files

### After Training
```
best_model_20epochs.pth              # ← Load this for inference
training_plots/
├── training_summary_20epochs_*.png  # ← Training metrics
├── training_history_20epochs.json   # ← Raw data
└── training_report_20epochs.txt     # ← Text summary
```

### After Inference
```
predictions_20epochs/
└── Case_*_prediction.nii.gz  # ← Segmentation results
```

### After Visualization
```
visualizations_20epochs/
└── *.png  # ← 2D slice comparisons
```

## Monitoring Training

### Check GPU Usage
```powershell
# Windows (NVIDIA GPU)
nvidia-smi

# Continuous monitoring
nvidia-smi -l 1  # Update every 1 second
```

### View Training Progress
Watch console output for:
- ✅ Epoch number
- ✅ Train/Val loss
- ✅ Dice scores
- ✅ Learning rate
- ⚠️  OOM errors

### Check Training Plots
Open `training_plots/training_summary_<N>epochs_*.png` to see:
- Loss curves
- Dice score progression
- Per-class performance

## Evaluation Metrics

### Dice Coefficient
```
Dice = 2 * |Prediction ∩ Ground Truth| / (|Prediction| + |Ground Truth|)
```
- **1.0**: Perfect match
- **≥0.7**: Good (project requirement)
- **0.5**: Moderate overlap
- **<0.5**: Poor

### Class Performance
| Dice Range | Quality |
|------------|---------|
| 0.9 - 1.0  | Excellent |
| 0.8 - 0.9  | Very Good |
| 0.7 - 0.8  | Good ✅ |
| 0.6 - 0.7  | Acceptable |
| < 0.6      | Poor ❌ |

## Troubleshooting Quick Fixes

### 🔴 Out of Memory
```python
# In config.py
'train_batch_size': 1,
'base_filters': 16,
```

### 🔴 Training Too Slow
```python
# In config.py
'use_amp': True,
'num_workers': 8,
```

### 🔴 Model Not Converging
```python
# In config.py
'learning_rate': 5e-4,  # Try lower
'num_epochs': 50,       # Train longer
```

### 🔴 Data Not Found
```python
# In config.py - Use raw string and absolute path
'data_root': r"C:\Users\YourName\Desktop\data",
```

## Expected Runtime

| Hardware | Epoch Time | Total (20 epochs) |
|----------|------------|-------------------|
| RTX 3080 | ~8-10 min  | ~2.5-3 hours      |
| RTX 4090 | ~4-5 min   | ~1.5-2 hours      |
| CPU      | ~60-90 min | ~20-30 hours ❌   |

## Validation Checklist

Before training:
- [ ] Updated `data_root` in `config.py`
- [ ] Ran `check_data.py` successfully
- [ ] GPU detected: `torch.cuda.is_available() == True`
- [ ] Sufficient disk space (~5GB for outputs)
- [ ] Virtual environment activated

During training:
- [ ] No OOM errors
- [ ] Loss decreasing
- [ ] Dice scores improving
- [ ] Learning rate visible in logs

After training:
- [ ] `best_model_*epochs.pth` exists
- [ ] Training plots generated
- [ ] All classes have Dice ≥ 0.7

## Useful Python Snippets

### Load and Inspect Checkpoint
```python
import torch

checkpoint = torch.load('best_model_20epochs.pth', weights_only=False)
print(f"Epoch: {checkpoint['epoch']}")
print(f"Best Dice: {checkpoint['best_dice']:.4f}")
print(f"Class Dice: {checkpoint['class_dice_scores']}")
```

### Quick Inference Test
```python
from predict import predict
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3D(in_channels=1, num_classes=6).to(device)
checkpoint = torch.load('best_model_20epochs.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

pred, affine, shape = predict(model, 'path/to/test_image.nii.gz', device)
print(f"Prediction shape: {pred.shape}")
```

### Count Model Parameters
```python
model = UNet3D(base_filters=24)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

## Command Cheat Sheet

```powershell
# Environment
.\venv\Scripts\activate              # Activate venv (Windows)
deactivate                           # Deactivate venv

# Package Management
pip list                             # List installed packages
pip freeze > requirements.txt        # Export dependencies
pip install package_name             # Install single package

# GPU Monitoring
nvidia-smi                           # GPU status
nvidia-smi -l 1                      # Continuous monitoring

# File Operations
dir training_plots                   # List training outputs (Windows)
# ls training_plots                  # List training outputs (Linux/Mac)
del best_model_*.pth                 # Delete old models (Windows)
# rm best_model_*.pth                # Delete old models (Linux/Mac)

# Python
python --version                     # Check Python version
python -c "import torch; print(torch.__version__)"  # Check PyTorch
python -m pip install --upgrade pip  # Update pip
```

## Emergency Recovery

### Training Crashed Mid-Way
Unfortunately, current implementation doesn't support resume. To continue:
1. Note which epoch crashed
2. Reduce memory usage (batch_size, base_filters)
3. Start training again (will restart from epoch 0)

### Lost Best Model File
- Check if backup exists in training_plots/
- Re-run training (takes time but works)
- Use last saved checkpoint if any

### Visualization Won't Generate
```python
# In visualize.py, check if prediction directory exists:
import os
print(os.path.exists('predictions_20epochs'))  # Should be True
```

## Performance Optimization Checklist

- [x] `use_amp: True` (40% memory saved)
- [x] `num_workers: 4-8` (faster data loading)
- [ ] `use_preload: True` (if you have 32GB+ RAM)
- [ ] SSD instead of HDD for dataset
- [ ] Close other GPU applications
- [ ] Latest NVIDIA drivers installed
- [ ] cudnn.benchmark enabled (automatic in code)

## Additional Resources

- **PyTorch Docs**: https://pytorch.org/docs/
- **MONAI Docs**: https://docs.monai.io/
- **Medical Image Analysis**: https://paperswithcode.com/task/medical-image-segmentation

---

**Quick Help**: Check README.md for detailed documentation
