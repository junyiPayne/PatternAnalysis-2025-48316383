import nibabel as nib
import numpy as np
import glob
import os
from config import CONFIG  

def check_paired_data(data_root, img_subdir, label_subdir):
    """Check if images and labels are correctly paired and display basic info"""

    # Image and label paths
    img_dir = os.path.join(data_root, img_subdir)
    label_dir = os.path.join(data_root, label_subdir)

    # Get all files
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.nii*")))
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.nii*")))
    
    print(f"Found {len(img_files)} images")
    print(f"Found {len(label_files)} labels\n")

    # Check filename matching
    img_names = [os.path.basename(f) for f in img_files[:5]]
    label_names = [os.path.basename(f) for f in label_files[:5]]
    
    print("First 5 image files:")
    for name in img_names:
        print(f"  {name}")
    
    print("\nFirst 5 label files:")
    for name in label_names:
        print(f"  {name}")

    # Check the content of the first label file
    if label_files:
        print(f"\n{'='*60}")
        print(f"Checking label file: {os.path.basename(label_files[0])}")
        print(f"{'='*60}")
        
        label_nifti = nib.load(label_files[0])
        label_data = label_nifti.get_fdata()
        
        print(f"Shape: {label_data.shape}")
        print(f"Data type: {label_data.dtype}")
        print(f"Unique values: {np.unique(label_data)}")
        print(f"Number of classes: {len(np.unique(label_data))}")

        # Count the number of voxels for each class
        print("\nClass distribution:")
        for class_id in np.unique(label_data):
            count = np.sum(label_data == class_id)
            percentage = 100 * count / label_data.size
            print(f"  Class {int(class_id)}: {count:,} voxels ({percentage:.2f}%)")
    
    # Check the content of the first image file
    if img_files:
        print(f"\n{'='*60}")
        print(f"Checking image file: {os.path.basename(img_files[0])}")
        print(f"{'='*60}")
        
        img_nifti = nib.load(img_files[0])
        img_data = img_nifti.get_fdata()
        
        print(f"Shape: {img_data.shape}")
        print(f"Data type: {img_data.dtype}")
        print(f"Min value: {img_data.min():.2f}")
        print(f"Max value: {img_data.max():.2f}")
        print(f"Mean value: {img_data.mean():.2f}")

if __name__ == "__main__":
    # read config
    data_root = CONFIG['data_root']
    img_subdir = CONFIG['img_subdir']  
    label_subdir = CONFIG['label_subdir']

    check_paired_data(data_root, img_subdir, label_subdir)