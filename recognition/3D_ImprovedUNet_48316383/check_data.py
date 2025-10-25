import nibabel as nib
import numpy as np
import glob
import os

def check_paired_data(data_root):
    """检查图像和标签的配对情况"""
    
    # 图像和标签路径
    img_dir = os.path.join(data_root, "HipMRI_study_complete_release_v1", "semantic_MRs_anon")
    label_dir = os.path.join(data_root, "HipMRI_study_complete_release_v1", "semantic_labels_anon")
    
    # 获取所有文件
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.nii*")))
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.nii*")))
    
    print(f"Found {len(img_files)} images")
    print(f"Found {len(label_files)} labels\n")
    
    # 检查文件名匹配
    img_names = [os.path.basename(f) for f in img_files[:5]]
    label_names = [os.path.basename(f) for f in label_files[:5]]
    
    print("First 5 image files:")
    for name in img_names:
        print(f"  {name}")
    
    print("\nFirst 5 label files:")
    for name in label_names:
        print(f"  {name}")
    
    # 检查第一个标签文件的内容
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
        
        # 统计每个类别的体素数
        print("\nClass distribution:")
        for class_id in np.unique(label_data):
            count = np.sum(label_data == class_id)
            percentage = 100 * count / label_data.size
            print(f"  Class {int(class_id)}: {count:,} voxels ({percentage:.2f}%)")
    
    # 检查对应的图像文件
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
    data_root = r"C:\Users\17561\Desktop\new 3710\data"
    check_paired_data(data_root)