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
    
    Args:
        model: trained UNet3D model
        image_path: path to .nii or .nii.gz file
        device: 'cuda' or 'cpu'
        target_size: tuple (D,H,W) for downsampling, must match training size
    
    Returns:
        prediction: label map (D, H, W)
        affine: original affine matrix for saving
        original_shape: original image shape for upsampling back if needed
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

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    num_classes = 6
    base_filters = 32
    target_size = (128, 128, 64)
    
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
    test_img_dir = os.path.join(data_root, "semantic_MRs_anon")
    
    test_images = sorted(glob.glob(os.path.join(test_img_dir, "*.nii.gz")))
    
    if not test_images:
        print(f"No test images found in {test_img_dir}")
        print("Please adjust test_img_dir path in the script.")
        exit(1)
    
    print(f"Found {len(test_images)} images to process\n")
    
    output_dir = Path("./predictions")
    output_dir.mkdir(exist_ok=True)
    
    for img_path in test_images:
        img_name = os.path.basename(img_path)
        print(f"Processing: {img_name}")
        
        try:
            prediction, affine, orig_shape = predict(model, img_path, device, target_size=target_size)
            
            output_name = img_name.replace('.nii.gz', '_pred.nii.gz').replace('.nii', '_pred.nii')
            output_path = output_dir / output_name
            
            nifti_pred = nib.Nifti1Image(prediction.astype(np.uint8), affine)
            nib.save(nifti_pred, str(output_path))
            
            unique_labels = np.unique(prediction)
            print(f"  Saved: {output_path.name}")
            print(f"  Shape: {prediction.shape}, Labels present: {unique_labels.tolist()}")
            
        except Exception as e:
            print(f"  ERROR processing {img_name}: {e}")
            continue
    
    print(f"\nDone! All predictions saved to {output_dir.absolute()}")