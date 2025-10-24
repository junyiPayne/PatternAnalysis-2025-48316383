# dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import glob
import os
from tqdm import tqdm
from scipy.ndimage import zoom

def to_channels(arr: np.ndarray, num_classes=6, dtype=np.uint8) -> np.ndarray:
    """将标签数组转换为 One-Hot 编码 - 修复版"""
    res = np.zeros((num_classes,) + arr.shape, dtype=dtype)  # (C, D, H, W)
    for c in range(num_classes):
        res[c] = (arr == c).astype(dtype)
    return res

def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, orient=False, early_stop=False, num_classes=6):
    """预加载所有图像到内存"""
    affines = []
    num = len(imageNames)
    
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:,:,:,0]
    
    if categorical:
        first_case = to_channels(first_case, num_classes=num_classes, dtype=dtype)
        channels, rows, cols, depth = first_case.shape
        images = np.zeros((num, channels, rows, cols, depth), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)
    
    for i, inName in enumerate(tqdm(imageNames, desc="Loading data")):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine
        
        if len(inImage.shape) == 4:
            inImage = inImage[:,:,:,0]
        
        inImage = inImage[:,:,:depth]
        inImage = inImage.astype(dtype)
        
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        
        if categorical:
            inImage = to_channels(inImage, num_classes=num_classes, dtype=dtype)
            images[i] = inImage
        else:
            images[i] = inImage
        
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
        self.num_classes = num_classes
        self.transform = transform
        self.target_size = target_size
        self.preload = preload
        
        base_dir = os.path.join(data_root, "HipMRI_study_complete_release_v1")
        img_dir = os.path.join(base_dir, "semantic_MRs_anon")
        label_dir = os.path.join(base_dir, "semantic_labels_anon")
        
        all_images = sorted(glob.glob(os.path.join(img_dir, "*.nii*")))
        all_labels = sorted(glob.glob(os.path.join(label_dir, "*.nii*")))
        
        print(f"Found {len(all_images)} images and {len(all_labels)} labels")
        
        np.random.seed(seed)
        indices = np.random.permutation(len(all_images))
        
        train_end = int(split_ratio[0] * len(indices))
        val_end = train_end + int(split_ratio[1] * len(indices))
        
        if split == 'train':
            selected_indices = indices[:train_end]
        elif split == 'val':
            selected_indices = indices[train_end:val_end]
        else:
            selected_indices = indices[val_end:]
        
        self.image_paths = [all_images[i] for i in selected_indices]
        self.label_paths = [all_labels[i] for i in selected_indices]
        
        print(f"{split.upper()} set: {len(self.image_paths)} samples")
        
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
            image = self.images[idx]  # (D, H, W)
            label = self.labels[idx]  # (C, D, H, W)
            
            if self.target_size:
                zoom_factors = [t / s for t, s in zip(self.target_size, image.shape)]
                image = zoom(image, zoom_factors, order=1)
                zoom_factors_4d = [1] + zoom_factors  # 保持通道维度
                label = zoom(label, zoom_factors_4d, order=0)
            
            image = torch.from_numpy(image).unsqueeze(0).float()
            label = torch.from_numpy(label).float()
            
        else:
            img_nifti = nib.load(self.image_paths[idx])
            image = img_nifti.get_fdata(caching='unchanged').astype(np.float32)
            
            label_nifti = nib.load(self.label_paths[idx])
            label = label_nifti.get_fdata(caching='unchanged').astype(np.uint8)
            
            if len(image.shape) == 4:
                image = image[:, :, :, 0]
            if len(label.shape) == 4:
                label = label[:, :, :, 0]
            
            if self.target_size:
                zoom_factors = [t / o for t, o in zip(self.target_size, image.shape)]
                image = zoom(image, zoom_factors, order=1)
                label = zoom(label, zoom_factors, order=0)
            
            label = np.clip(label, 0, self.num_classes - 1)
            image = (image - image.mean()) / (image.std() + 1e-8)
            
            # 统一使用修复后的 one-hot 编码
            label_onehot = to_channels(label, num_classes=self.num_classes, dtype=np.uint8)
            
            image = torch.from_numpy(image).unsqueeze(0).float()
            label = torch.from_numpy(label_onehot).float()
        
        if self.transform:
            image, label = self.transform(image, label)
        
        return image, label