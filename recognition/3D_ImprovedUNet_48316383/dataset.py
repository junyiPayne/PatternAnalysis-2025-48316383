import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import glob
import os
from tqdm import tqdm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from utils import to_channels, plot_class_distribution

# MONAI imports for data augmentation
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandAffined,
    RandScaleIntensityd,
    EnsureTyped,
)

def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, orient=False, early_stop=False, num_classes=6):
    """preload 3D data from given file names."""
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
            data_root (str): data root dictionary
            split (str): 'train', 'val', or 'test'
            num_classes (int): number of segmentation classes
            target_size (tuple): target image size
            preload (bool): whether to preload data
            transform (callable): data augmentation function
            seed (int): random seed
            split_ratio (tuple): training, validation, and test split ratios
        """
        self.num_classes = num_classes
        self.transform = transform  
        self.target_size = target_size
        self.preload = preload

        # validate split_ratio 
        assert sum(split_ratio) == 1.0, "split_ratio must sum to 1"
        assert all(0 <= r <= 1 for r in split_ratio), "split_ratio values must be between 0 and 1"

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
        
        self.class_weights = self._compute_class_weights()  

    def _compute_class_weights(self, method='inverse_sqrt_frequency'):
        """
        calculate class weights based on voxel frequency.
        """
        print("\nStarting class weight computation...")
        class_counts = torch.zeros(self.num_classes)
        total_voxels = 0

        original_transform = self.transform
        self.transform = None

        for idx in tqdm(range(len(self)), desc="Counting class distribution"):
            _, label = self.__getitem__(idx)
            label_tensor = torch.from_numpy(label) if isinstance(label, np.ndarray) else label

            # Count the number of voxels for each class
            for c in range(self.num_classes):
                count = (label_tensor[c] > 0.5).sum().item()
                class_counts[c] += count
                total_voxels += count

        # Restore original transform
        self.transform = original_transform

        # Save class distribution information
        self.class_counts = class_counts

        # Compute class weights
        class_frequencies = class_counts / total_voxels
        if method == 'inverse_sqrt_frequency':
            weights = 1.0 / torch.sqrt(class_frequencies + 1e-6)
        else:  # inverse_frequency
            weights = 1.0 / (class_frequencies + 1e-6)

        # Normalize weights
        weights = weights / weights.mean()
        weights = torch.clamp(weights, min=0.1, max=15.0)

        print("\nClass Statistics:")
        for c in range(self.num_classes):
            percentage = (class_counts[c] / total_voxels) * 100
            print(f"Class {c}: {class_counts[c]:,.1f} voxels ({percentage:.2f}%)")

        print("\nComputed Weights:")
        for c in range(self.num_classes):
            print(f"Class {c}: {weights[c]:.4f}")

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
                zoom_factors_4d = [1] + zoom_factors  
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
            
            
            label_onehot = to_channels(label, num_classes=self.num_classes, dtype=np.uint8)
            
            image = torch.from_numpy(image).unsqueeze(0).float()
            label = torch.from_numpy(label_onehot).float()

        # Data augmentation
        if self.transform:
            image, label = self.transform(image, label)
        
        return image, label

class MONAIAugmentation:
    """
    Augmentation based on MONAI:
    - Random flipping (3 axes)
    - Adding Gaussian noise
    - Random intensity scaling

    Note: Due to the asymmetric shape of target_size=(128,128,64), avoid using rotation to maintain shape consistency
    """
    def __init__(self, flip_prob=0.5, noise_std=0.05):
        """
        Initialize MONAI data augmentation

        Args:
            flip_prob: Probability of flipping
            noise_std: Standard deviation of Gaussian noise
        """
        self.flip_prob = flip_prob
        self.noise_std = noise_std

        # Build MONAI augmentation pipeline - only use transformations that strictly maintain shape
        self.transforms = Compose([
            # Random flipping (along D, H, W axes) - maintain shape
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=flip_prob),  # D axis
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=flip_prob),  # H axis
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=flip_prob),  # W axis

            # Adding Gaussian noise (only apply to image)
            RandGaussianNoised(keys=["image"], prob=1.0, std=noise_std),

            # Random intensity scaling (only apply to image) - multiplicative transform, maintain shape
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        ])
    
    def __call__(self, image, label):
        """
        Apply MONAI augmentation to image and label

        Args:
            image: Image tensor of shape (C, D, H, W), e.g., (1, 128, 128, 64)
            label: Label tensor of shape (C, D, H, W) (one-hot encoded), e.g., (6, 128, 128, 64)

        Returns:
            Augmented (image, label) with strict shape preservation
        """
        # Record original shapes for verification
        orig_image_shape = image.shape
        orig_label_shape = label.shape
        
        # MONAI needs dict input
        data_dict = {
            "image": image,
            "label": label
        }

        # Apply augmentations
        augmented = self.transforms(data_dict)

        # Get augmented data
        aug_image = augmented["image"]
        aug_label = augmented["label"]

        # Convert to standard PyTorch tensor (MONAI may return MetaTensor)
        if not isinstance(aug_image, torch.Tensor):
            aug_image = torch.as_tensor(aug_image)
        if not isinstance(aug_label, torch.Tensor):
            aug_label = torch.as_tensor(aug_label)
        
        # Verify shape preservation
        assert aug_image.shape == orig_image_shape, \
            f"Image shape changed: {orig_image_shape} -> {aug_image.shape}"
        assert aug_label.shape == orig_label_shape, \
            f"Label shape changed: {orig_label_shape} -> {aug_label.shape}"
            
        return aug_image, aug_label


# MONAI-BASED DATA LOADING (Alternative to Prostate3DDataset)
from monai.data import Dataset as MONAIDataset, CacheDataset, DataLoader
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Resized,
    ScaleIntensityRanged, AsDiscreted
)


def get_monai_data_dicts(data_root, img_subdir, label_subdir, split='train', 
                         split_ratio=(0.7, 0.15, 0.15), seed=42):
    """
    Generate MONAI data dictionaries for efficient data loading
    
    Args:
        data_root (str): Root directory containing data
        img_subdir (str): Subdirectory for images
        label_subdir (str): Subdirectory for labels
        split (str): 'train', 'val', or 'test'
        split_ratio (tuple): Train/Val/Test split ratios
        seed (int): Random seed for reproducibility
        
    Returns:
        list: List of dictionaries with 'image' and 'label' keys
    """
    # Get all image and label paths
    img_dir = os.path.join(data_root, img_subdir)
    label_dir = os.path.join(data_root, label_subdir)
    
    all_images = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    all_labels = sorted(glob.glob(os.path.join(label_dir, "*.nii.gz")))
    
    assert len(all_images) == len(all_labels), \
        f"Number of images ({len(all_images)}) != number of labels ({len(all_labels)})"
    
    # Split data
    np.random.seed(seed)
    indices = np.random.permutation(len(all_images))
    
    train_end = int(split_ratio[0] * len(indices))
    val_end = train_end + int(split_ratio[1] * len(indices))
    
    if split == 'train':
        selected_indices = indices[:train_end]
    elif split == 'val':
        selected_indices = indices[train_end:val_end]
    elif split == 'test':
        selected_indices = indices[val_end:]
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
    
    # Create data dictionaries
    data_dicts = [
        {
            'image': all_images[i],
            'label': all_labels[i]
        }
        for i in selected_indices
    ]
    
    print(f"📁 {split.upper()} split: {len(data_dicts)} samples")
    
    return data_dicts


def get_monai_train_transforms(target_size=(128, 128, 64), num_classes=6, 
                                flip_prob=0.5, noise_std=0.05):
    """
    Get MONAI training data augmentation transforms
    
    Args:
        target_size (tuple): Target image size (H, W, D)
        num_classes (int): Number of segmentation classes
        flip_prob (float): Probability of random flip
        noise_std (float): Standard deviation of Gaussian noise
        
    Returns:
        Compose: MONAI transform composition
    """
    return Compose([
        # Loading and basic preprocessing
        LoadImaged(keys=['image', 'label'], image_only=False),
        EnsureChannelFirstd(keys=['image', 'label']),
        
        # Resize to target size
        Resized(keys=['image', 'label'], 
               spatial_size=target_size,
               mode=['trilinear', 'nearest']),
        
        # Intensity normalization (image only)
        ScaleIntensityRanged(
            keys=['image'],
            a_min=None, a_max=None,  # Auto-detect min/max
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        
        # Data augmentation
        RandFlipd(keys=['image', 'label'], spatial_axis=0, prob=flip_prob),
        RandFlipd(keys=['image', 'label'], spatial_axis=1, prob=flip_prob),
        RandFlipd(keys=['image', 'label'], spatial_axis=2, prob=flip_prob),
        RandGaussianNoised(keys=['image'], prob=1.0, std=noise_std),
        RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.3),
        
        # Convert label to one-hot encoding
        AsDiscreted(keys=['label'], to_onehot=num_classes),
        
        # Final type conversion
        EnsureTyped(keys=['image', 'label'], dtype=torch.float32),
    ])


def get_monai_val_transforms(target_size=(128, 128, 64), num_classes=6):
    """
    Get MONAI validation/test transforms (no augmentation)
    
    Args:
        target_size (tuple): Target image size (H, W, D)
        num_classes (int): Number of segmentation classes
        
    Returns:
        Compose: MONAI transform composition
    """
    return Compose([
        # Loading and basic preprocessing
        LoadImaged(keys=['image', 'label'], image_only=False),
        EnsureChannelFirstd(keys=['image', 'label']),
        
        # Resize to target size
        Resized(keys=['image', 'label'], 
               spatial_size=target_size,
               mode=['trilinear', 'nearest']),
        
        # Intensity normalization (image only)
        ScaleIntensityRanged(
            keys=['image'],
            a_min=None, a_max=None,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        
        # Convert label to one-hot encoding
        AsDiscreted(keys=['label'], to_onehot=num_classes),
        
        # Final type conversion
        EnsureTyped(keys=['image', 'label'], dtype=torch.float32),
    ])


def create_monai_dataloaders(config, use_cache=True):
    """
    Create MONAI data loaders for training, validation, and testing
    
    This is the main function to use MONAI-based data loading.
    
    Args:
        config (dict): Configuration dictionary from config.py
        use_cache (bool): Whether to use CacheDataset for faster loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
        
    Example:
        >>> from config import CONFIG
        >>> train_loader, val_loader, test_loader = create_monai_dataloaders(CONFIG)
    """
    # Extract configuration
    data_root = config['data_root']
    img_subdir = config['img_subdir']
    label_subdir = config['label_subdir']
    target_size = config['target_size']
    num_classes = config['num_classes']
    seed = config.get('seed', 42)
    split_ratio = config.get('split_ratio', (0.7, 0.15, 0.15))
    
    # Augmentation parameters
    aug_config = config.get('augmentation', {})
    flip_prob = aug_config.get('flip_prob', 0.5)
    noise_std = aug_config.get('noise_std', 0.05)
    
    # Data loader parameters
    train_batch_size = config.get('train_batch_size', 2)
    val_batch_size = config.get('val_batch_size', 1)
    cache_rate = config.get('cache_rate', 0.5)
    
    print("="*80)
    print("Creating MONAI Data Loaders")
    print("="*80)
    
    # Get data dictionaries
    train_dicts = get_monai_data_dicts(data_root, img_subdir, label_subdir, 
                                       split='train', split_ratio=split_ratio, seed=seed)
    val_dicts = get_monai_data_dicts(data_root, img_subdir, label_subdir, 
                                     split='val', split_ratio=split_ratio, seed=seed)
    test_dicts = get_monai_data_dicts(data_root, img_subdir, label_subdir, 
                                      split='test', split_ratio=split_ratio, seed=seed)
    
    # Create transforms
    train_transforms = get_monai_train_transforms(target_size, num_classes, flip_prob, noise_std)
    val_transforms = get_monai_val_transforms(target_size, num_classes)
    
    # Create datasets
    DatasetClass = CacheDataset if use_cache else MONAIDataset
    
    if use_cache:
        print(f"\n📦 Using CacheDataset (cache_rate={cache_rate})")
        train_ds = CacheDataset(
            data=train_dicts,
            transform=train_transforms,
            cache_rate=cache_rate,
            num_workers=4
        )
        val_ds = CacheDataset(
            data=val_dicts,
            transform=val_transforms,
            cache_rate=cache_rate,
            num_workers=4
        )
        test_ds = CacheDataset(
            data=test_dicts,
            transform=val_transforms,
            cache_rate=cache_rate,
            num_workers=4
        )
    else:
        print("\n📦 Using standard MONAI Dataset")
        train_ds = MONAIDataset(data=train_dicts, transform=train_transforms)
        val_ds = MONAIDataset(data=val_dicts, transform=val_transforms)
        test_ds = MONAIDataset(data=test_dicts, transform=val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for CacheDataset to avoid multiprocessing issues
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"\n✅ MONAI data loaders created successfully!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print("="*80 + "\n")
    
    return train_loader, val_loader, test_loader


def compute_class_weights_monai(data_loader, num_classes, device='cpu'):
    """
    Compute class weights from a MONAI data loader
    
    Args:
        data_loader: MONAI DataLoader
        num_classes (int): Number of classes
        device (str): Device for computation
        
    Returns:
        torch.Tensor: Class weights
    """
    print("Computing class weights from MONAI loader...")
    
    class_counts = torch.zeros(num_classes, device=device)
    
    for batch in tqdm(data_loader, desc="Computing weights"):
        labels = batch['label'].to(device)  # (B, C, D, H, W)
        
        # Sum over batch and spatial dimensions
        batch_counts = labels.sum(dim=(0, 2, 3, 4))  # (C,)
        class_counts += batch_counts
    
    # Compute weights using inverse square root frequency
    class_frequencies = class_counts / class_counts.sum()
    class_weights = 1.0 / torch.sqrt(class_frequencies + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print(f"✅ Class weights computed:")
    for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
        print(f"   Class {i}: count={int(count):,}, weight={weight:.4f}")
    
    return class_weights


