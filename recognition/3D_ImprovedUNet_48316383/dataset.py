# dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import glob
import os
from tqdm import tqdm
from scipy.ndimage import zoom, rotate
import matplotlib.pyplot as plt

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
    def __init__(self, data_root, split='train', num_classes=6, target_size=(128, 128, 64), 
                 preload=False, transform=None, seed=42, split_ratio=(0.7, 0.15, 0.15)):
        """
        参数:
            data_root (str): 数据根目录
            split (str): 'train', 'val', 或 'test'
            num_classes (int): 类别数量
            target_size (tuple): 目标图像大小
            preload (bool): 是否预加载数据
            transform (callable): 数据增强函数
            seed (int): 随机种子
            split_ratio (tuple): 训练集、验证集、测试集的比例
        """
        self.num_classes = num_classes
        self.transform = transform  # 数据增强逻辑
        self.target_size = target_size
        self.preload = preload
        
        # 验证 split_ratio 的合法性
        assert sum(split_ratio) == 1.0, "split_ratio 必须和为 1"
        assert all(0 <= r <= 1 for r in split_ratio), "split_ratio 的每个值必须在 0 和 1 之间"
        
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
        
        self.class_weights = self._compute_class_weights()  # 添加这行

    def _compute_class_weights(self, method='inverse_sqrt_frequency'):
        """
        计算类别权重并统计类别分布
        """
        print("\n开始计算类别权重...")
        class_counts = torch.zeros(self.num_classes)
        total_voxels = 0

        # 临时禁用数据增强，以便统计类别分布时不应用增强
        original_transform = self.transform
        self.transform = None

        for idx in tqdm(range(len(self)), desc="统计类别分布"):
            _, label = self.__getitem__(idx)
            label_tensor = torch.from_numpy(label) if isinstance(label, np.ndarray) else label

            # 统计每个类别的体素数量
            for c in range(self.num_classes):
                count = (label_tensor[c] > 0.5).sum().item()
                class_counts[c] += count
                total_voxels += count

        # 恢复原始的 transform
        self.transform = original_transform

        # 保存类别分布信息
        self.class_counts = class_counts

        # 计算类别权重
        class_frequencies = class_counts / total_voxels
        if method == 'inverse_sqrt_frequency':
            weights = 1.0 / torch.sqrt(class_frequencies + 1e-6)
        else:  # inverse_frequency
            weights = 1.0 / (class_frequencies + 1e-6)

        # 归一化权重
        weights = weights / weights.mean()
        weights = torch.clamp(weights, min=0.1, max=15.0)

        print("\n类别统计:")
        for c in range(self.num_classes):
            percentage = (class_counts[c] / total_voxels) * 100
            print(f"类别 {c}: {class_counts[c]:,.1f} 体素 ({percentage:.2f}%)")

        print("\n计算得到的权重:")
        for c in range(self.num_classes):
            print(f"类别 {c}: {weights[c]:.4f}")

        return weights

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
        
        # 数据增强逻辑
        if self.transform:
            image, label = self.transform(image, label)
        
        return image, label

class RandomAugmentation:
    """数据增强类"""
    def __init__(self, flip_prob=0.5, noise_std=0.05, rotate_angle=15):
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.rotate_angle = rotate_angle

    def __call__(self, image, label):
        # 随机翻转
        if np.random.rand() < self.flip_prob:
            image = torch.flip(image, dims=[2])  # 水平翻转
            label = torch.flip(label, dims=[2])
        
        if np.random.rand() < self.flip_prob:
            image = torch.flip(image, dims=[3])  # 垂直翻转
            label = torch.flip(label, dims=[3])
        
        # 添加高斯噪声
        if np.random.rand() < self.flip_prob:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
        
        # 随机旋转
        if self.rotate_angle > 0:
            angle = np.random.uniform(-self.rotate_angle, self.rotate_angle)
            axes = (2, 3)  # 在 (H, W) 平面上旋转
            image = torch.from_numpy(rotate(image.numpy(), angle, axes=axes, reshape=False, order=1))
            label = torch.from_numpy(rotate(label.numpy(), angle, axes=axes, reshape=False, order=0))
        
        return image, label

def plot_class_distribution(class_counts, save_path="class_distribution.png"):
    """
    绘制类别分布图并保存为图片
    Args:
        class_counts (torch.Tensor): 每个类别的体素数量
        save_path (str): 保存路径
    """
    num_classes = len(class_counts)
    plt.figure(figsize=(8, 6))
    plt.bar(range(num_classes), class_counts.cpu().numpy(), color='skyblue')
    plt.xlabel("Class Index")
    plt.ylabel("Voxel Count")
    plt.title("Class Distribution")
    plt.xticks(range(num_classes))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Class distribution plot saved to {save_path}")