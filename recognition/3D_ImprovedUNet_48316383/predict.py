"""
Inference script for 3D medical image segmentation.
Loads trained UNet3D model and generates predictions with same preprocessing as training.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import nibabel as nib
from modules import UNet3D
from scipy.ndimage import zoom
import glob
from pathlib import Path
from config import CONFIG  # 导入配置
from torch.amp import autocast  # 导入 AMP

def predict(model, image_path, device='cuda', target_size=None, use_amp=False):
    """
    Predict segmentation for a single 3D image with optional downsampling.
    
    Args:
        model: trained UNet3D model
        image_path: Input image path
        device: Device ('cuda' or 'cpu')
        target_size: Target size, if None use original size
        use_amp: Whether to use mixed precision inference

    Returns:
        prediction: Prediction result (original size)
        affine: NIfTI affine matrix
        original_shape: Original image shape
    """
    model.eval()

    # Load image
    img_nifti = nib.load(image_path)
    image = img_nifti.get_fdata(caching='unchanged').astype(np.float32)
    affine = img_nifti.affine
    original_shape = image.shape

    # Process 4D images (remove time dimension)
    if len(image.shape) == 4:
        image = image[:, :, :, 0]

    # Resize to target size if specified
    if target_size is not None:
        zoom_factors = [t / o for t, o in zip(target_size, image.shape)]
        image = zoom(image, zoom_factors, order=1)

    # Standardize (same as training)
    image = (image - image.mean()) / (image.std() + 1e-8)

    # Convert to tensor (1, 1, D, H, W)
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)

    # Inference
    with torch.no_grad():
        if use_amp and device != 'cpu':
            with autocast(device_type='cuda'):
                output = model(image_tensor)
        else:
            output = model(image_tensor)
        
        # Softmax + Argmax
        output = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).cpu().numpy()[0]

    # Restore original size
    if target_size is not None:
        zoom_factors_back = [o / t for o, t in zip(original_shape, target_size)]
        prediction = zoom(prediction, zoom_factors_back, order=0)
    
    return prediction, affine, original_shape

def dice_coefficient_np(pred, target, num_classes):
    """Calculate Dice coefficient for numpy arrays."""
    dice_scores = np.zeros(num_classes)
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union == 0:
            dice_scores[c] = 1.0 if intersection == 0 else 0.0
        else:
            dice_scores[c] = (2.0 * intersection) / union
    return dice_scores

def get_test_split(all_images, all_labels, split_ratio=(0.7, 0.15, 0.15), seed=42):
    """Get test set indices matching dataset.py split logic."""
    np.random.seed(seed)
    indices = np.random.permutation(len(all_images))
    
    train_end = int(split_ratio[0] * len(indices))
    val_end = train_end + int(split_ratio[1] * len(indices))
    
    test_indices = indices[val_end:]
    test_images = [all_images[i] for i in test_indices]
    test_labels = [all_labels[i] for i in test_indices]
    
    return test_images, test_labels

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ============================================================
    # 配置区域：修改这里来测试不同的模型
    # ============================================================
    # 可选: "best_model_10epochs.pth", "best_model_20epochs.pth", "best_model_100epochs.pth"
    checkpoint_path = "C:\\Users\\17561\\Desktop\\new 3710\\best_model_100epochs.pth"  # 👈 修改这里切换模型
    # ============================================================
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract epoch number from checkpoint for directory naming
    trained_epochs = checkpoint.get('epoch', 'unknown')
    print(f"✅ Loaded checkpoint from epoch {trained_epochs}")
    print(f"   Best validation Dice: {checkpoint.get('best_dice', 'N/A'):.4f}")

    # Load config from checkpoint (prefer checkpoint config to ensure consistency with training)
    if 'config' in checkpoint:
        loaded_config = checkpoint['config']
        print("\n📋 Using config from checkpoint:")
        print(f"   num_classes: {loaded_config['num_classes']}")
        print(f"   base_filters: {loaded_config['base_filters']}")
        print(f"   target_size: {loaded_config.get('target_size', 'Not specified')}")
        
        num_classes = loaded_config['num_classes']
        base_filters = loaded_config['base_filters']
        in_channels = loaded_config.get('in_channels', 1)
        target_size = loaded_config.get('target_size', None)
        use_amp = loaded_config.get('use_amp', False)
    else:
        # if there is no config in checkpoint, use current CONFIG
        print("\n⚠️  No config in checkpoint, using current CONFIG")
        num_classes = CONFIG['num_classes']
        base_filters = CONFIG['base_filters']
        in_channels = CONFIG.get('in_channels', 1)
        target_size = CONFIG.get('target_size', None)
        use_amp = CONFIG.get('use_amp', False)
    
    print(f"   use_amp: {use_amp}\n")

    # user-defined model
    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=base_filters
    ).to(device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded successfully")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Load test data paths（use CONFIG，because data paths may change）
    data_root = CONFIG['data_root']
    img_dir = os.path.join(data_root, CONFIG['img_subdir'])
    label_dir = os.path.join(data_root, CONFIG['label_subdir'])
    
    all_images = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    all_labels = sorted(glob.glob(os.path.join(label_dir, "*.nii.gz")))

    # get test set（same with dataset.py）
    seed = CONFIG.get('seed', 42)  
    test_images, test_labels = get_test_split(all_images, all_labels, seed=seed)
    
    print(f"Total images: {len(all_images)}")
    print(f"Test set size: {len(test_images)} (15% of total)\n")
    
    # Create output directory with epoch suffix for tracking different training runs
    if trained_epochs != 'unknown':
        output_dir = Path(f"./predictions_{trained_epochs}epochs")
    else:
        output_dir = Path("./predictions")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Predictions will be saved to: {output_dir}\n")
    
    all_dice_scores = []
    
    for img_path, label_path in zip(test_images, test_labels):
        img_name = os.path.basename(img_path)
        print(f"Processing: {img_name}")
        
        try:
            # generate prediction results（use same settings as training）
            prediction, affine, orig_shape = predict(
                model, img_path, device, 
                target_size=target_size, 
                use_amp=use_amp
            )

            # Load ground truth labels
            label_nifti = nib.load(label_path)
            gt_label = label_nifti.get_fdata(caching='unchanged').astype(np.uint8)
            if len(gt_label.shape) == 4:
                gt_label = gt_label[:, :, :, 0]
            gt_label = np.clip(gt_label, 0, num_classes - 1)

            # calculate Dice scores
            dice_scores = dice_coefficient_np(prediction, gt_label, num_classes)
            all_dice_scores.append(dice_scores)
            
            # keep results
            output_name = img_name.replace('.nii.gz', '_pred.nii.gz').replace('.nii', '_pred.nii')
            output_path = output_dir / output_name
            
            nifti_pred = nib.Nifti1Image(prediction.astype(np.uint8), affine)
            nib.save(nifti_pred, str(output_path))
            
            unique_labels = np.unique(prediction)
            mean_dice = np.mean(dice_scores[1:])  # remove background class
            print(f"  Saved: {output_path.name}")
            print(f"  Dice per class: {[f'{d:.4f}' for d in dice_scores]}")
            print(f"  Mean Dice (excl. bg): {mean_dice:.4f}\n")
            
        except Exception as e:
            print(f"  ERROR processing {img_name}: {e}\n")
            continue
    
    # calculate overall statistics
    if all_dice_scores:
        all_dice_scores = np.array(all_dice_scores)
        mean_dice_per_class = np.mean(all_dice_scores, axis=0)
        std_dice_per_class = np.std(all_dice_scores, axis=0)
        mean_dice_overall = np.mean(mean_dice_per_class[1:])
        
        print("\n" + "="*60)
        print("TEST SET RESULTS")
        print("="*60)
        print(f"Number of test cases: {len(all_dice_scores)}")
        print(f"\nDice Score per Class (mean ± std):")
        for c in range(num_classes):
            print(f"  Class {c}: {mean_dice_per_class[c]:.4f} ± {std_dice_per_class[c]:.4f}")
        print(f"\nMean Dice (excluding background): {mean_dice_overall:.4f}")

        # Check if requirements are met
        min_dice_non_bg = np.min(mean_dice_per_class[1:])
        print(f"Minimum Dice (excluding background): {min_dice_non_bg:.4f}")
        
        if np.all(mean_dice_per_class[1:] >= 0.7):
            print("\n✅ SUCCESS: All classes have Dice ≥ 0.7!")
        else:
            failing_classes = np.where(mean_dice_per_class[1:] < 0.7)[0] + 1
            print(f"\n❌ REQUIREMENT NOT MET: Classes {failing_classes.tolist()} have Dice < 0.7")
        
        print("="*60)
        
        # Save test results to file for reporting
        if trained_epochs != 'unknown':
            result_filename = f"test_results_{trained_epochs}epochs.txt"
        else:
            result_filename = "test_results.txt"
        
        result_path = Path(result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("TEST SET RESULTS (Final Evaluation)\n")
            f.write("="*60 + "\n\n")
            f.write(f"Model: {checkpoint_path}\n")
            f.write(f"Trained Epochs: {trained_epochs}\n")
            f.write(f"Number of Test Cases: {len(all_dice_scores)}\n")
            f.write(f"Number of Classes: {num_classes}\n\n")
            
            f.write("-"*60 + "\n")
            f.write("PER-CLASS DICE SCORES (Mean ± Std)\n")
            f.write("-"*60 + "\n")
            for c in range(num_classes):
                f.write(f"Class {c}: {mean_dice_per_class[c]:.4f} ± {std_dice_per_class[c]:.4f}\n")
            
            f.write("\n" + "-"*60 + "\n")
            f.write("SUMMARY METRICS\n")
            f.write("-"*60 + "\n")
            f.write(f"Mean Dice (excluding background): {mean_dice_overall:.4f}\n")
            f.write(f"Minimum Dice (excluding background): {min_dice_non_bg:.4f}\n")
            
            f.write("\n" + "-"*60 + "\n")
            f.write("REQUIREMENT CHECK\n")
            f.write("-"*60 + "\n")
            f.write("Project Requirement: All classes Dice ≥ 0.7\n\n")
            
            if np.all(mean_dice_per_class[1:] >= 0.7):
                f.write("✅ SUCCESS: All classes meet requirement (Dice ≥ 0.7)\n")
            else:
                failing_classes = np.where(mean_dice_per_class[1:] < 0.7)[0] + 1
                f.write(f"❌ REQUIREMENT NOT MET\n")
                f.write(f"Failing classes: {failing_classes.tolist()}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("DETAILED PER-CASE RESULTS\n")
            f.write("="*60 + "\n\n")
            
            for idx, (img_path, dice_scores) in enumerate(zip(test_images, all_dice_scores)):
                img_name = os.path.basename(img_path)
                f.write(f"Case {idx+1}: {img_name}\n")
                f.write(f"  Dice scores: {[f'{d:.4f}' for d in dice_scores]}\n")
                f.write(f"  Mean (excl. bg): {np.mean(dice_scores[1:]):.4f}\n\n")
        
        print(f"\n📄 Test results saved to: {result_path.absolute()}")
    
    print(f"\nAll predictions saved to {output_dir.absolute()}")