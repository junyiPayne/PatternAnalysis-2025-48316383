# dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import glob
import os
from tqdm import tqdm
from scipy.ndimage import zoom

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """将标签数组转换为 One-Hot 编码（你之前的函数）"""
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, orient=False, early_stop=False, num_classes=6):
    """
    你之前的 load_data_3D 函数（简化版）
    Load all images into memory at once
    """
    affines = []
    num = len(imageNames)
    
    # 获取第一个样本的尺寸
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:,:,:,0]
    
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)
    
    for i, inName in enumerate(tqdm(imageNames, desc="Loading data")):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine
        
        if len(inImage.shape) == 4:
            inImage = inImage[:,:,:,0]
        
        inImage = inImage[:,:,:depth]  # clip slices
        inImage = inImage.astype(dtype)
        
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i,:inImage.shape[0],:inImage.shape[1],:inImage.shape[2],:inImage.shape[3]] = inImage
        else:
            images[i,:inImage.shape[0],:inImage.shape[1],:inImage.shape[2]] = inImage
        
        affines.append(affine)
        if i > 20 and early_stop:
            break
    
    if getAffines:
        return images, affines
    else:
        return images

class Prostate3DDataset(Dataset):
    def __init__(self, data_root, split='train', split_ratio=(0.7, 0.15, 0.15), 
                 num_classes=6, transform=None, seed=42, target_size=(128, 128, 64),
                 preload=False):
        """
        Args:
            preload: True = 使用你之前的 load_data_3D 预加载所有数据到内存
                     False = 按需加载（当前方式）
        """
        self.num_classes = num_classes
        self.transform = transform
        self.target_size = target_size
        self.preload = preload
        
        # 设置路径
        base_dir = os.path.join(data_root, "HipMRI_study_complete_release_v1")
        img_dir = os.path.join(base_dir, "semantic_MRs_anon")
        label_dir = os.path.join(base_dir, "semantic_labels_anon")
        
        # 获取所有文件
        all_images = sorted(glob.glob(os.path.join(img_dir, "*.nii*")))
        all_labels = sorted(glob.glob(os.path.join(label_dir, "*.nii*")))
        
        print(f"Found {len(all_images)} images and {len(all_labels)} labels")
        
        # 数据集划分
        np.random.seed(seed)
        indices = np.random.permutation(len(all_images))
        
        train_end = int(split_ratio[0] * len(indices))
        val_end = train_end + int(split_ratio[1] * len(indices))
        
        if split == 'train':
            selected_indices = indices[:train_end]
        elif split == 'val':
            selected_indices = indices[train_end:val_end]
        else:  # test
            selected_indices = indices[val_end:]
        
        self.image_paths = [all_images[i] for i in selected_indices]
        self.label_paths = [all_labels[i] for i in selected_indices]
        
        print(f"{split.upper()} set: {len(self.image_paths)} samples")
        
        # 如果 preload=True，使用你之前的函数预加载
        if self.preload:
            print("Preloading images...")
            self.images = load_data_3D(self.image_paths, normImage=True, 
                                      categorical=False, dtype=np.float32)
            print("Preloading labels...")
            self.labels = load_data_3D(self.label_paths, normImage=False, 
                                      categorical=True, dtype=np.uint8,
                                      num_classes=num_classes)
            print("Preload complete!")
        else:
            self.images = None
            self.labels = None
            if target_size:
                print(f"Downsampling to: {target_size}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.preload:
            # 使用预加载的数据
            image = self.images[idx]  # (D, H, W)
            label = self.labels[idx]  # (D, H, W, C)
            
            # 下采样（如果需要）
            if self.target_size:
                zoom_factors = [t / s for t, s in zip(self.target_size, image.shape)]
                image = zoom(image, zoom_factors, order=1)
                zoom_factors_4d = zoom_factors + [1]  # 保持通道维度
                label = zoom(label, zoom_factors_4d, order=0)
            
            # 转换为 tensor
            image = torch.from_numpy(image).unsqueeze(0).float()  # (1, D, H, W)
            label = torch.from_numpy(label).permute(3, 0, 1, 2).float()  # (C, D, H, W)
            
        else:
            # 按需加载（原来的方式）
            img_nifti = nib.load(self.image_paths[idx])
            image = img_nifti.get_fdata(caching='unchanged').astype(np.float32)
            
            label_nifti = nib.load(self.label_paths[idx])
            label = label_nifti.get_fdata(caching='unchanged').astype(np.uint8)
            
            # Remove extra dimensions if 4D
            if len(image.shape) == 4:
                image = image[:, :, :, 0]
            if len(label.shape) == 4:
                label = label[:, :, :, 0]
            
            # Downsample
            if self.target_size:
                zoom_factors = [t / o for t, o in zip(self.target_size, image.shape)]
                image = zoom(image, zoom_factors, order=1)
                label = zoom(label, zoom_factors, order=0)
            
            # Clip label values
            label = np.clip(label, 0, self.num_classes - 1)
            
            # Normalize image
            image = (image - image.mean()) / (image.std() + 1e-8)
            
            # Convert label to one-hot
            label_onehot = np.zeros((self.num_classes,) + label.shape, dtype=np.uint8)
            for c in range(self.num_classes):
                label_onehot[c] = (label == c).astype(np.uint8)
            
            # Convert to tensors
            image = torch.from_numpy(image).unsqueeze(0).float()
            label = torch.from_numpy(label_onehot).float()
        
        if self.transform:
            image, label = self.transform(image, label)
        
        return image, label