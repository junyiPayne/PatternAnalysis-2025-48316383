import torch
import numpy as np
import nibabel as nib
from modules import UNet3D
from dataset import to_channels
import glob
import os

def predict(model, image_path, device='cuda'):
    """Predict segmentation for a single 3D image"""
    model.eval()
    
    # Load image
    img_nifti = nib.load(image_path)
    image = img_nifti.get_fdata(caching='unchanged').astype(np.float32)
    affine = img_nifti.affine
    
    if len(image.shape) == 4:
        image = image[:, :, :, 0]
    
    # Normalize
    image = (image - image.mean()) / (image.std() + 1e-8)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, D, H, W)
    
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).cpu().numpy()[0]  # (D, H, W)
    
    return prediction, affine

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = UNet3D(in_channels=1, num_classes=5).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Predict on test set
    test_img_dir = "./data/test/images"
    test_images = glob.glob(os.path.join(test_img_dir, "*.nii.gz"))
    
    os.makedirs("./predictions", exist_ok=True)
    
    for img_path in test_images:
        prediction, affine = predict(model, img_path, device)
        
        # Save prediction
        output_name = os.path.basename(img_path).replace('.nii.gz', '_pred.nii.gz')
        output_path = os.path.join("./predictions", output_name)
        
        nifti_pred = nib.Nifti1Image(prediction.astype(np.uint8), affine)
        nib.save(nifti_pred, output_path)
        
        print(f"Saved prediction: {output_path}")