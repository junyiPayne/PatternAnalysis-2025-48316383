"""
Visualization script for 3D segmentation results.
Generates 2D slice comparisons (original image, ground truth, prediction) for README display.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import glob

def normalize_image(image):
    """Normalize image to [0, 1] for display."""
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image

def create_colormap(num_classes=6):
    """Create color map for segmentation labels."""
    colors = [
        [0, 0, 0],        # Class 0: Background (black)
        [1, 0, 0],        # Class 1: Red
        [0, 1, 0],        # Class 2: Green
        [0, 0, 1],        # Class 3: Blue
        [1, 1, 0],        # Class 4: Yellow
        [1, 0, 1],        # Class 5: Magenta
    ]
    return np.array(colors[:num_classes])

def label_to_rgb(label, num_classes=6, alpha=0.5):
    """Convert label map to RGB image with transparency."""
    colormap = create_colormap(num_classes)
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    for c in range(num_classes):
        mask = (label == c)
        rgb[mask] = colormap[c]
    
    return rgb

def visualize_slice(image_slice, gt_slice, pred_slice, slice_idx, axis_name, 
                   output_path, num_classes=6, class_names=None):
    """
    Visualize a single 2D slice comparison.
    
    Args:
        image_slice: 2D array of original image
        gt_slice: 2D array of ground truth labels
        pred_slice: 2D array of predicted labels
        slice_idx: slice index in volume
        axis_name: 'Axial', 'Coronal', or 'Sagittal'
        output_path: path to save the figure
        num_classes: number of segmentation classes
        class_names: optional list of class names for legend
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Normalize image for display
    img_normalized = normalize_image(image_slice)
    
    # 1. Original image
    axes[0].imshow(img_normalized, cmap='gray')
    axes[0].set_title(f'Original Image\n{axis_name} Slice {slice_idx}', fontsize=12)
    axes[0].axis('off')
    
    # 2. Ground truth overlay
    axes[1].imshow(img_normalized, cmap='gray')
    gt_rgb = label_to_rgb(gt_slice, num_classes, alpha=0.5)
    axes[1].imshow(gt_rgb, alpha=0.5)
    axes[1].set_title('Ground Truth Overlay', fontsize=12)
    axes[1].axis('off')
    
    # 3. Prediction overlay
    axes[2].imshow(img_normalized, cmap='gray')
    pred_rgb = label_to_rgb(pred_slice, num_classes, alpha=0.5)
    axes[2].imshow(pred_rgb, alpha=0.5)
    axes[2].set_title('Prediction Overlay', fontsize=12)
    axes[2].axis('off')
    
    # 4. Side-by-side comparison (GT | Pred)
    comparison = np.concatenate([gt_rgb, pred_rgb], axis=1)
    axes[3].imshow(comparison)
    axes[3].axvline(x=gt_slice.shape[1], color='white', linestyle='--', linewidth=2)
    axes[3].set_title('GT (left) | Prediction (right)', fontsize=12)
    axes[3].axis('off')
    
    # Add legend
    colormap = create_colormap(num_classes)
    patches = [mpatches.Patch(color=colormap[i], label=class_names[i]) 
               for i in range(num_classes)]
    fig.legend(handles=patches, loc='lower center', ncol=num_classes, 
              fontsize=10, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def visualize_volume(image_path, gt_path, pred_path, output_dir, case_name,
                    num_classes=6, class_names=None, slices_per_axis=3):
    """
    Generate visualization for multiple slices along different axes.
    
    Args:
        image_path: path to original image .nii.gz
        gt_path: path to ground truth label .nii.gz
        pred_path: path to prediction .nii.gz
        output_dir: directory to save visualizations
        case_name: name for this case (for file naming)
        num_classes: number of classes
        class_names: optional list of class names
        slices_per_axis: number of slices to extract per axis
    """
    # Load volumes
    image = nib.load(image_path).get_fdata(caching='unchanged').astype(np.float32)
    gt_label = nib.load(gt_path).get_fdata(caching='unchanged').astype(np.uint8)
    pred_label = nib.load(pred_path).get_fdata(caching='unchanged').astype(np.uint8)
    
    # Handle 4D volumes
    if len(image.shape) == 4:
        image = image[:, :, :, 0]
    if len(gt_label.shape) == 4:
        gt_label = gt_label[:, :, :, 0]
    if len(pred_label.shape) == 4:
        pred_label = pred_label[:, :, :, 0]
    
    D, H, W = image.shape
    
    # Select slice indices
    slice_positions = np.linspace(0.25, 0.75, slices_per_axis)
    
    # Axial slices (along Z/depth axis)
    print(f"Generating axial slices for {case_name}...")
    for i, pos in enumerate(slice_positions):
        slice_idx = int(pos * D)
        img_slice = image[slice_idx, :, :]
        gt_slice = gt_label[slice_idx, :, :]
        pred_slice = pred_label[slice_idx, :, :]
        
        output_path = output_dir / f"{case_name}_axial_slice_{slice_idx:03d}.png"
        visualize_slice(img_slice, gt_slice, pred_slice, slice_idx, 'Axial',
                       output_path, num_classes, class_names)
    
    # Coronal slices (along Y/height axis)
    print(f"Generating coronal slices for {case_name}...")
    for i, pos in enumerate(slice_positions):
        slice_idx = int(pos * H)
        img_slice = image[:, slice_idx, :]
        gt_slice = gt_label[:, slice_idx, :]
        pred_slice = pred_label[:, slice_idx, :]
        
        output_path = output_dir / f"{case_name}_coronal_slice_{slice_idx:03d}.png"
        visualize_slice(img_slice, gt_slice, pred_slice, slice_idx, 'Coronal',
                       output_path, num_classes, class_names)
    
    # Sagittal slices (along X/width axis)
    print(f"Generating sagittal slices for {case_name}...")
    for i, pos in enumerate(slice_positions):
        slice_idx = int(pos * W)
        img_slice = image[:, :, slice_idx]
        gt_slice = gt_label[:, :, slice_idx]
        pred_slice = pred_label[:, :, slice_idx]
        
        output_path = output_dir / f"{case_name}_sagittal_slice_{slice_idx:03d}.png"
        visualize_slice(img_slice, gt_slice, pred_slice, slice_idx, 'Sagittal',
                       output_path, num_classes, class_names)

def get_test_split(all_images, all_labels, split_ratio=(0.7, 0.15, 0.15), seed=42):
    """Get test set indices matching dataset.py split logic."""
    np.random.seed(seed)
    indices = np.random.permutation(len(all_images))
    
    train_end = int(split_ratio[0] * len(indices))
    val_end = train_end + int(split_ratio[1] * len(indices))
    
    test_indices = indices[val_end:]
    test_images = [all_images[i] for i in test_indices]
    test_labels = [all_labels[i] for i in test_indices]
    
    return test_images, test_labels, test_indices

if __name__ == "__main__":
    # Configuration
    data_root = r"C:\Users\17561\Desktop\new 3710\data\HipMRI_study_complete_release_v1"
    img_dir = os.path.join(data_root, "semantic_MRs_anon")
    label_dir = os.path.join(data_root, "semantic_labels_anon")
    
    # Auto-detect prediction directory with epoch suffix
    # First, try to find directories matching pattern "predictions_*epochs"
    pred_dirs = sorted(glob.glob("./predictions_*epochs"))
    
    if pred_dirs:
        # Use the most recent prediction directory (last in sorted list)
        pred_dir = Path(pred_dirs[-1])
        # Extract epoch suffix from directory name (e.g., "predictions_20epochs" -> "20epochs")
        epoch_suffix = pred_dir.name.replace("predictions_", "")
        print(f"📁 Found prediction directory: {pred_dir}")
        print(f"📊 Detected training epochs: {epoch_suffix}\n")
    else:
        # Fallback to default predictions directory
        pred_dir = Path("./predictions")
        epoch_suffix = None
        print(f"📁 Using default prediction directory: {pred_dir}")
        print(f"⚠️  No epoch information detected in directory name\n")
    
    # Create output directory with epoch suffix if available
    if epoch_suffix:
        output_dir = Path(f"./visualizations_{epoch_suffix}")
    else:
        output_dir = Path("./visualizations")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Visualizations will be saved to: {output_dir}\n")
    
    num_classes = 6
    class_names = ['Background', 'Prostate', 'Bladder', 'Rectum', 'Femur_L', 'Femur_R']
    seed = 42
    
    # Get test set
    all_images = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    all_labels = sorted(glob.glob(os.path.join(label_dir, "*.nii.gz")))
    test_images, test_labels, test_indices = get_test_split(all_images, all_labels, seed=seed)
    
    print(f"Total images: {len(all_images)}")
    print(f"Test set size: {len(test_images)}\n")
    
    # Visualize first 3 test cases (or all if you want)
    num_cases_to_visualize = min(3, len(test_images))
    
    for i in range(num_cases_to_visualize):
        img_path = test_images[i]
        gt_path = test_labels[i]
        
        img_name = os.path.basename(img_path)
        case_name = img_name.replace('.nii.gz', '').replace('.nii', '')
        pred_name = img_name.replace('.nii.gz', '_pred.nii.gz').replace('.nii', '_pred.nii')
        pred_path = pred_dir / pred_name
        
        if not pred_path.exists():
            print(f"Warning: Prediction not found for {img_name}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Visualizing case {i+1}/{num_cases_to_visualize}: {case_name}")
        print(f"{'='*60}")
        
        visualize_volume(
            image_path=img_path,
            gt_path=gt_path,
            pred_path=str(pred_path),
            output_dir=output_dir,
            case_name=case_name,
            num_classes=num_classes,
            class_names=class_names,
            slices_per_axis=3  # Generate 3 slices per axis (total 9 images per case)
        )
    
    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"All images saved to: {output_dir.absolute()}")
    print(f"{'='*60}")
    print("\nYou can now include these images in your README.md:")
    print("Example markdown syntax:")
    print("```markdown")
    print("## Segmentation Results")
    print("")
    print("### Case 1: [Case Name]")
    print(f"![Axial Slice](./visualizations/[case_name]_axial_slice_XXX.png)")
    print(f"![Coronal Slice](./visualizations/[case_name]_coronal_slice_XXX.png)")
    print("```")