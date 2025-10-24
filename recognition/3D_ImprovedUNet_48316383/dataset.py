# dataset.py

import numpy as np
import nibabel as nib
from tqdm import tqdm
import glob
import os
# 如果使用 PyTorch，您可能需要导入 torch 和 torch.utils.data.Dataset
# 如果使用 TensorFlow，您可能需要导入 tensorflow 和 tf.data.Dataset

# --- 辅助函数：将标签转换为 One-Hot 编码 (参考报告附录 B to_channels 函数) ---
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """将包含多个整数类别的标签数组转换为多通道 One-Hot 编码"""
    # 报告中的示例代码假设类别从 0 开始且连续
    channels = np.unique(arr)
    # 创建一个在最后一维增加通道数的新数组
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    
    # 遍历所有唯一的标签值
    for c in channels:
        C = int(c)
        # 将原始数组中等于 c 的位置在新数组的相应通道上设为 1
        res[arr == c, C] = 1 
        
    return res

# --- 核心函数：加载 3D Nifti 数据 (基于报告附录 B load_data_3D 示例) ---
def load_data_3D(image_names, norm_image=False, categorical=False, dtype=np.float32, 
                 get_affines=False, orient=False, early_stop=False):
    """
    从 Nifti 文件名列表中加载 3D 医疗影像数据。
    注意：您需要实现完整的错误检查和内存管理逻辑。
    """
    
    # --- 1. 获取固定尺寸（Find Fixed Size） ---
    num = len(image_names)
    nifti_image = nib.load(image_names[0])
    first_case = nifti_image.get_fdata(caching='unchanged')

    # 处理可能的 4D 数据（例如移除额外维度），并确定最终形状
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0] # 移除第 4 维 [cite: 394, 399]

    if categorical:
        # 如果是标签，转换为 One-Hot 编码来确定通道数 [cite: 410]
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        # 如果是图像，只有一个通道
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    # --- 2. 循环加载所有数据 ---
    for i, in_name in enumerate(tqdm(image_names)):
        nifti_image = nib.load(in_name)
        in_image = nifti_image.get_fdata(caching='unchanged') # 读取数据 [cite: 423]
        
        # 处理可能的 4D 数据和尺寸裁剪
        if len(in_image.shape) == 4:
            in_image = in_image[:, :, :, 0] # 移除额外维度 [cite: 426]
            # in_image = in_image[:, :, :depth] # 剪裁切片以匹配固定深度 [cite: 427]

        in_image = in_image.astype(dtype)

        # 归一化处理（如果需要）
        if norm_image:
            # 报告中的归一化示例是 z-score 标准化 [cite: 441, 449]
            in_image = (in_image - in_image.mean()) / in_image.std()

        # 标签转换为 One-Hot 编码（如果需要）
        if categorical:
            in_image = to_channels(in_image, dtype=dtype) # [cite: 450]
            # 存储 5D 数组：(num, rows, cols, depth, channels)
            images[i, :] = in_image
        else:
            # 存储 4D 数组：(num, rows, cols, depth)
            images[i, :] = in_image
        
        # ... 其他逻辑（如保存仿射矩阵 affines, 提前停止 early_stop）[cite: 360, 466]

    return images

# --- Data Loader 类（例如 PyTorch 的 Dataset） ---
# 建议您封装上述函数，实现一个高效的 DataLoader 类
class Prostate3DDataset():
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform # 用于数据增强

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 实际加载逻辑应调用 load_data_3D 或类似功能
        # 为了效率，可以只加载单个文件并进行预处理
        image = nib.load(self.image_paths[idx]).get_fdata()
        label = nib.load(self.label_paths[idx]).get_fdata()

        # ... 在这里进行必要的预处理、裁剪、标准化和 One-Hot 编码 ...
        
        if self.transform:
            image, label = self.transform(image, label)

        # 调整轴序以适应您的深度学习框架 (例如 PyTorch: C, D, H, W)
        return image, label