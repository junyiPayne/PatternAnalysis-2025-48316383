# Project Workflow Diagram

## Complete Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ENVIRONMENT SETUP                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Install Python 3.8+                                             │
│  2. Create virtual environment: python -m venv venv                 │
│  3. Activate: .\venv\Scripts\activate                               │
│  4. Install PyTorch: pip install torch torchvision                  │
│  5. Install dependencies: pip install -r requirements.txt           │
│  6. Verify: python -c "import torch; print(torch.cuda.is_available())"│
│                                                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     CONFIGURATION                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Edit config.py:                                                    │
│  ✏️  data_root = r"C:\path\to\your\data"                           │
│  ✏️  num_epochs = 20                                               │
│  ✏️  train_batch_size = 2 (or 1 if OOM)                            │
│  ✏️  base_filters = 24 (or 16 if OOM)                              │
│                                                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 1: DATA VALIDATION                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Command: python check_data.py                                      │
│                                                                      │
│  Checks:                                                             │
│  ✅ File count matching (images vs labels)                          │
│  ✅ Data format (NIfTI .nii.gz)                                     │
│  ✅ Label values (0-5 for 6 classes)                                │
│  ✅ Class distribution                                              │
│  ✅ Data shapes and types                                           │
│                                                                      │
│  Output:                                                             │
│  - Console: Data statistics                                         │
│  - Verification: Dataset is ready                                   │
│                                                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 2: MODEL TRAINING                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Command: python train.py                                           │
│                                                                      │
│  Process:                                                            │
│  1. Load data (70% train / 15% val / 15% test)                     │
│  2. Initialize UNet3D model                                         │
│  3. Setup FocalDiceLoss, optimizer, scheduler                       │
│  4. Training loop (with AMP, dynamic weights, early stopping)      │
│     ├─ Forward pass                                                 │
│     ├─ Compute loss                                                 │
│     ├─ Backward pass                                                │
│     ├─ Update weights                                               │
│     └─ Validate every epoch                                         │
│  5. Track best model (in memory)                                    │
│  6. Save best model at end                                          │
│  7. Generate visualizations and reports                             │
│                                                                      │
│  Output Files:                                                       │
│  📄 best_model_20epochs.pth                                         │
│  📊 training_plots/                                                  │
│     ├── training_summary_20epochs_<timestamp>.png                   │
│     ├── training_history_20epochs.json                              │
│     └── training_report_20epochs.txt                                │
│                                                                      │
│  Console Output:                                                     │
│  - Epoch progress bars                                              │
│  - Train/Val loss and Dice scores                                   │
│  - Learning rate updates                                            │
│  - Best model notifications                                         │
│  - Early stopping alerts (if triggered)                             │
│                                                                      │
│  Expected Time: 2-3 hours (RTX 3080)                                │
│                                                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 3: INFERENCE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Command: python predict.py                                         │
│                                                                      │
│  Process:                                                            │
│  1. Load best_model_<N>epochs.pth                                   │
│  2. Extract trained epochs for naming                               │
│  3. Load test set (same split as training)                          │
│  4. For each test image:                                            │
│     ├─ Preprocess (normalize, resize)                               │
│     ├─ Run inference with model                                     │
│     ├─ Apply softmax + argmax                                       │
│     ├─ Resize back to original size                                 │
│     ├─ Calculate Dice scores                                        │
│     └─ Save as NIfTI file                                           │
│  5. Compute overall statistics                                      │
│                                                                      │
│  Output Files:                                                       │
│  📁 predictions_20epochs/                                            │
│     ├── Case_001_Week0_LFOV_prediction.nii.gz                       │
│     ├── Case_002_Week0_LFOV_prediction.nii.gz                       │
│     └── ... (all test cases)                                        │
│                                                                      │
│  Console Output:                                                     │
│  - Per-case Dice scores                                             │
│  - Mean Dice across all test cases                                  │
│  - Per-class Dice statistics                                        │
│                                                                      │
│  Expected Time: 10-15 minutes                                       │
│                                                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 4: VISUALIZATION                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Command: python visualize.py                                       │
│                                                                      │
│  Process:                                                            │
│  1. Auto-detect predictions_<N>epochs/ directory                    │
│  2. Load first 3 test cases (configurable)                          │
│  3. For each case:                                                  │
│     └─ For each axis (axial, coronal, sagittal):                    │
│        ├─ Extract 3 representative slices                           │
│        ├─ Create 4-panel comparison:                                │
│        │  • Original grayscale image                                │
│        │  • Ground truth overlay                                    │
│        │  • Prediction overlay                                      │
│        │  • Side-by-side GT | Prediction                            │
│        └─ Save as PNG with color legend                             │
│                                                                      │
│  Output Files:                                                       │
│  📁 visualizations_20epochs/                                         │
│     ├── Case_001_axial_slice_032.png                                │
│     ├── Case_001_coronal_slice_064.png                              │
│     ├── Case_001_sagittal_slice_064.png                             │
│     └── ... (9 images per case)                                     │
│                                                                      │
│  Features:                                                           │
│  - Color-coded segmentation classes                                 │
│  - Transparent overlays for easy comparison                         │
│  - High-resolution PNG (DPI=150)                                    │
│  - Class legend on each image                                       │
│                                                                      │
│  Expected Time: 5-10 minutes                                        │
│                                                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RESULTS ANALYSIS                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Review Generated Files:                                             │
│                                                                      │
│  1. Training Metrics:                                               │
│     📊 training_plots/training_summary_20epochs_*.png               │
│        - Loss curves                                                │
│        - Dice score progression                                     │
│        - Per-class performance                                      │
│        - Worst-K class tracking                                     │
│        - Learning rate schedule                                     │
│                                                                      │
│  2. Prediction Quality:                                             │
│     📁 predictions_20epochs/*.nii.gz                                │
│        - Can be opened in ITK-SNAP or 3D Slicer                     │
│        - Compare with ground truth                                  │
│                                                                      │
│  3. Visual Comparisons:                                             │
│     🎨 visualizations_20epochs/*.png                                │
│        - Qualitative assessment                                     │
│        - Identify strengths and weaknesses                          │
│        - Use in reports/presentations                               │
│                                                                      │
│  4. Quantitative Metrics:                                           │
│     📄 training_report_20epochs.txt                                 │
│        - Final Dice scores                                          │
│        - Per-class statistics                                       │
│        - Training summary                                           │
│                                                                      │
│  Expected Results:                                                   │
│  ✅ All classes achieve Dice ≥ 0.7                                  │
│  ✅ Mean Dice (excl. background) ≈ 0.83-0.85                        │
│  ✅ Smooth segmentation boundaries                                  │
│  ✅ No major false positives                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
                    ┌─────────────────┐
                    │  Raw NIfTI Data │
                    │  (.nii.gz files)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Preprocessing  │
                    │  - Normalize    │
                    │  - Resize       │
                    │  - One-hot      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼────────┐     │     ┌────────▼────────┐
     │  Training Set   │     │     │   Test Set      │
     │     (70%)       │     │     │     (15%)       │
     └────────┬────────┘     │     └────────┬────────┘
              │              │              │
              │     ┌────────▼────────┐     │
              │     │ Validation Set  │     │
              │     │     (15%)       │     │
              │     └────────┬────────┘     │
              │              │              │
     ┌────────▼──────────────▼────────┐     │
     │      Training Loop              │     │
     │  ┌──────────────────────────┐  │     │
     │  │  Forward Pass            │  │     │
     │  │  ↓                       │  │     │
     │  │  Loss Calculation        │  │     │
     │  │  ↓                       │  │     │
     │  │  Backward Pass           │  │     │
     │  │  ↓                       │  │     │
     │  │  Weight Update           │  │     │
     │  └──────────────────────────┘  │     │
     │              ↓                  │     │
     │      Validation Check           │     │
     │              ↓                  │     │
     │    Dynamic Weight Adjust        │     │
     │              ↓                  │     │
     │      Early Stopping?            │     │
     └────────┬────────────────────────┘     │
              │                              │
              ▼                              │
     ┌──────────────────┐                    │
     │   Best Model     │                    │
     │  (best_model.pth)│                    │
     └────────┬─────────┘                    │
              │                              │
              └──────────────┬───────────────┘
                             │
                    ┌────────▼────────┐
                    │   Inference     │
                    │   (predict.py)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Predictions    │
                    │  (.nii.gz)      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Visualization  │
                    │  (visualize.py) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  PNG Images     │
                    │  (for reports)  │
                    └─────────────────┘
```

## File Dependencies

```
config.py ────────┐
                  ├─→ train.py ──→ best_model.pth
                  │      │
modules.py ───────┤      ├─→ training_visualizer.py
                  │      │         │
dataset.py ───────┤      │         └─→ training_plots/*.png
                  │      │
utils.py ─────────┘      └─→ (saves checkpoint)
                              │
                              ▼
                         predict.py ──→ predictions/*.nii.gz
                              │                 │
                              └─────────────────┘
                                               │
                                               ▼
                                          visualize.py
                                               │
                                               ▼
                                    visualizations/*.png
```

## Memory Usage at Each Stage

```
┌──────────────────────┬──────────────┬─────────────────┐
│ Stage                │ GPU Memory   │ Disk Space      │
├──────────────────────┼──────────────┼─────────────────┤
│ Model Loading        │ ~1.5 GB      │ -               │
│ Training (batch=2)   │ ~8-10 GB     │ -               │
│ Saving Checkpoint    │ -            │ ~500 MB         │
│ Inference            │ ~2-3 GB      │ ~2 GB (outputs) │
│ Visualization        │ Minimal      │ ~100 MB (PNGs)  │
└──────────────────────┴──────────────┴─────────────────┘
```

## Time Breakdown (RTX 3080)

```
┌──────────────────────┬──────────────────┐
│ Stage                │ Estimated Time   │
├──────────────────────┼──────────────────┤
│ Environment Setup    │ 10-15 minutes    │
│ Data Validation      │ 1-2 minutes      │
│ Training (20 epochs) │ 2.5-3 hours      │
│ Inference            │ 10-15 minutes    │
│ Visualization        │ 5-10 minutes     │
├──────────────────────┼──────────────────┤
│ Total                │ ~3-4 hours       │
└──────────────────────┴──────────────────┘
```

## Quality Checkpoints

At each stage, verify:

**After Environment Setup:**
- [ ] `torch.cuda.is_available() == True`
- [ ] All packages imported without errors
- [ ] GPU detected with `nvidia-smi`

**After Data Validation:**
- [ ] Image count matches label count
- [ ] No file loading errors
- [ ] Class distribution reasonable

**During Training:**
- [ ] Loss decreasing over epochs
- [ ] Dice scores improving
- [ ] No OOM errors
- [ ] Checkpoints being saved

**After Training:**
- [ ] Best model file exists
- [ ] Training plots generated
- [ ] All classes achieve Dice ≥ 0.7
- [ ] No overfitting (val loss stable)

**After Inference:**
- [ ] Predictions directory created
- [ ] All test cases processed
- [ ] Dice scores calculated
- [ ] NIfTI files valid

**After Visualization:**
- [ ] PNG images generated
- [ ] Overlays look correct
- [ ] Colors match legend
- [ ] No blank images

## Emergency Procedures

**If training crashes:**
1. Check GPU memory: `nvidia-smi`
2. Reduce batch_size or base_filters
3. Ensure AMP is enabled
4. Check error message in console

**If predictions look wrong:**
1. Verify correct checkpoint loaded
2. Check preprocessing matches training
3. Ensure target_size consistent
4. Validate with visualizations

**If visualization fails:**
1. Check predictions directory exists
2. Verify NIfTI files are valid
3. Ensure matplotlib working
4. Check disk space available
