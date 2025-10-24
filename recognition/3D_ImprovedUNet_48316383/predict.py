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

def predict(model, image_path, device='cuda', target_size=None):
    """
    Predict segmentation for a single 3D image with optional downsampling.
    """
    model.eval()
    
    img_nifti = nib.load(image_path)
    image = img_nifti.get_fdata(caching='unchanged').astype(np.float32)
    affine = img_nifti.affine
    original_shape = image.shape
    
    if len(image.shape) == 4:
        image = image[:, :, :, 0]
    
    if target_size is not None:
        zoom_factors = [t / o for t, o in zip(target_size, image.shape)]
        image = zoom(image, zoom_factors, order=1)
    
    image = (image - image.mean()) / (image.std() + 1e-8)
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
    
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
    
    # MUST match training configuration
    num_classes = 6
    base_filters = 32  # CHECK: match your train.py
    target_size = (128, 128, 64)  # CHECK: match your train.py
    seed = 42  # MUST match dataset.py
    
    model = UNet3D(in_channels=1, num_classes=num_classes, base_filters=base_filters).to(device)
    
    checkpoint_path = 'best_model.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
        print(f"Best validation Dice: {checkpoint.get('best_dice', 'N/A'):.4f}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded model weights from {checkpoint_path}")
    
    data_root = r"C:\Users\17561\Desktop\new 3710\data\HipMRI_study_complete_release_v1"
    img_dir = os.path.join(data_root, "semantic_MRs_anon")
    label_dir = os.path.join(data_root, "semantic_labels_anon")
    
    all_images = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    all_labels = sorted(glob.glob(os.path.join(label_dir, "*.nii.gz")))
    
    # Get TEST SET ONLY (matching dataset.py split)
    test_images, test_labels = get_test_split(all_images, all_labels, seed=seed)
    
    print(f"Total images: {len(all_images)}")
    print(f"Test set size: {len(test_images)} (15% of total)\n")
    
    output_dir = Path("./predictions")
    output_dir.mkdir(exist_ok=True)
    
    all_dice_scores = []
    
    for img_path, label_path in zip(test_images, test_labels):
        img_name = os.path.basename(img_path)
        print(f"Processing: {img_name}")
        
        try:
            # Generate prediction
            prediction, affine, orig_shape = predict(model, img_path, device, target_size=target_size)
            
            # Load ground truth
            label_nifti = nib.load(label_path)
            gt_label = label_nifti.get_fdata(caching='unchanged').astype(np.uint8)
            if len(gt_label.shape) == 4:
                gt_label = gt_label[:, :, :, 0]
            gt_label = np.clip(gt_label, 0, num_classes - 1)
            
            # Calculate Dice scores
            dice_scores = dice_coefficient_np(prediction, gt_label, num_classes)
            all_dice_scores.append(dice_scores)
            
            # Save prediction
            output_name = img_name.replace('.nii.gz', '_pred.nii.gz').replace('.nii', '_pred.nii')
            output_path = output_dir / output_name
            
            nifti_pred = nib.Nifti1Image(prediction.astype(np.uint8), affine)
            nib.save(nifti_pred, str(output_path))
            
            unique_labels = np.unique(prediction)
            mean_dice = np.mean(dice_scores[1:])  # Exclude background
            print(f"  Saved: {output_path.name}")
            print(f"  Dice per class: {[f'{d:.4f}' for d in dice_scores]}")
            print(f"  Mean Dice (excl. bg): {mean_dice:.4f}\n")
            
        except Exception as e:
            print(f"  ERROR processing {img_name}: {e}\n")
            continue
    
    # Calculate and display test set statistics
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
        
        # Check if requirement met
        min_dice_non_bg = np.min(mean_dice_per_class[1:])
        print(f"Minimum Dice (excluding background): {min_dice_non_bg:.4f}")
        
        if np.all(mean_dice_per_class[1:] >= 0.7):
            print("\n✅ SUCCESS: All classes have Dice ≥ 0.7!")
        else:
            failing_classes = np.where(mean_dice_per_class[1:] < 0.7)[0] + 1
            print(f"\n❌ REQUIREMENT NOT MET: Classes {failing_classes.tolist()} have Dice < 0.7")
        
        print("="*60)
    
    print(f"\nAll predictions saved to {output_dir.absolute()}")