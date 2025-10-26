CONFIG = {
    'num_classes': 6,  # 类别数
    'target_size': (128, 128, 64),  # 图像下采样尺寸
    'batch_size': 2,  # 训练批量大小
    'num_workers': 4,  # 数据加载线程数
    'base_filters': 16,  # UNet3D 的基础通道数
    'val_freq': 1,  # 每隔多少个 epoch 验证一次
    'grad_accum_steps': 1,  # 梯度累积步数
    'use_preload': False,  # 是否预加载数据
    'data_root': r"C:\Users\17561\Desktop\new 3710\data",  # 数据集路径
}