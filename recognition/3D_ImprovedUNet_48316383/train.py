import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from modules import UNet3D  # 确保导入的是标准 UNet3D
from dataset import Prostate3DDataset
import matplotlib.pyplot as plt

def init_weights_he(m):
    if isinstance(m, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if getattr(m, 'bias', None) is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, (torch.nn.BatchNorm3d, torch.nn.InstanceNorm3d)):
        if getattr(m, 'weight', None) is not None:
            torch.nn.init.ones_(m.weight)
        if getattr(m, 'bias', None) is not None:
            torch.nn.init.zeros_(m.bias)

# --- Weighted Dice Loss for imbalanced classes ---
class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, class_weights=None, weight_power=1.0):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.class_weights = class_weights
        self.weight_power = weight_power  # 控制权重的强度
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        
        # 计算每个类别的 Dice
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)
        
        if self.class_weights is not None:
            weights = self.class_weights.to(pred.device)
            # 使用 weight_power 来调整权重的影响
            weights = weights ** self.weight_power
            dice_per_class = dice_per_class * weights.unsqueeze(0)
        
        # 打印每个类别的 Dice 值，方便监控
        if not self.training:  # 在验证时打印
            for c in range(len(dice_per_class[0])):
                print(f"Class {c} Dice: {dice_per_class[0][c]:.4f}")
        
        return 1 - dice_per_class.mean()

# --- Focal Dice Loss ---
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, dice_weight=1.0, focal_weight=1.0, smooth=1.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, pred, target):
        # 确保 target 是 long 类型
        target = target.long()
        
        # 首先计算 Dice Loss
        pred_softmax = torch.softmax(pred, dim=1)
        
        # 创建 one-hot 编码
        batch_size, num_classes, depth, height, width = pred_softmax.size()
        target_one_hot = torch.zeros_like(pred_softmax)
        
        # 正确处理 target 的维度
        target = target.view(batch_size, 1, depth, height, width)
        target_one_hot.scatter_(1, target, 1)
        
        # Dice Loss
        intersection = (pred_softmax * target_one_hot).sum(dim=(2,3,4))
        union = pred_softmax.sum(dim=(2,3,4)) + target_one_hot.sum(dim=(2,3,4))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        if self.class_weights is not None:
            dice = dice * self.class_weights.to(dice.device)
        
        dice_loss = 1 - dice.mean()

        # Focal Loss
        pt = (pred_softmax * target_one_hot).sum(dim=1)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = -torch.log(pt + 1e-8) * focal_weight
        focal_loss = focal_loss.mean()

        # 组合损失
        total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        return total_loss

# --- Dice Coefficient Metric ---
def dice_coefficient(pred, target, num_classes):
    """Calculate Dice coefficient for each class"""
    pred = torch.argmax(pred, dim=1)  # (N, D, H, W)
    dice_scores = []
    
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target[:, c, :, :, :]).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union == 0:
            dice_scores.append(1.0 if intersection == 0 else 0.0)
        else:
            dice_scores.append((2. * intersection / union).item())
    
    return dice_scores

from torch.amp import autocast, GradScaler  # use new amp API

# --- Training Function (AMP, gradient accumulation, optional compile) ---
def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda', num_classes=6,
                val_freq=1, grad_accum_steps=1):
    # 修改初始化参数，使其与 DynamicWeightAdjuster 类定义匹配
    weight_adjuster = DynamicWeightAdjuster(
        initial_weights=train_loader.dataset.class_weights,
        window_size=3,
        target_dice=0.7,
        min_improvement=0.01
    )
    
    criterion = WeightedDiceLoss(
        smooth=1.0,
        class_weights=weight_adjuster.weights,
        weight_power=1.0
    )
    
    # 每个 epoch 打印类别权重的实际效果
    def print_class_stats(outputs, labels):
        with torch.no_grad():
            dice_scores = dice_coefficient(outputs, labels, num_classes)
            print(f"各类别 Dice: {[f'{d:.4f}' for d in dice_scores]}")
    
    # --- 其余训练代码保持不变 ---
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=7, factor=0.5)
    
    # Create GradScaler in a backwards-compatible way and decide whether to use AMP
    best_dice = 0.0
    amp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = False
    scaler = None
    try:
        # try new API first (may raise TypeError on older torch)
        scaler = GradScaler(device_type='cuda') if torch.cuda.is_available() else GradScaler()
        use_amp = True
    except TypeError:
        try:
            # fallback to older API
            scaler = GradScaler()
            use_amp = True
        except Exception:
            scaler = None
            use_amp = False
    if use_amp:
        print(f"AMP enabled (autocast device={amp_device})")
    else:
        print("AMP not available — running without mixed precision")
 
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f"Epoch {epoch+1}/{num_epochs} - Train", leave=False)
        optimizer.zero_grad()
        for batch_idx, (images, labels) in train_pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with autocast(device_type=amp_device):
                    outputs = model(images)
                    loss = criterion(outputs, labels) / grad_accum_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels) / grad_accum_steps
                loss.backward()
                if (batch_idx + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
 
            train_loss += loss.item() * grad_accum_steps  # scale back to real loss

            # per-batch metrics
            with torch.no_grad():
                batch_dice = dice_coefficient(outputs, labels, num_classes=num_classes)
            mean_batch_dice = np.mean(batch_dice[1:]) if num_classes > 1 else batch_dice[0]
            lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix(loss=f"{(loss.item()*grad_accum_steps):.4f}", mean_dice=f"{mean_batch_dice:.4f}", lr=f"{lr:.2e}")

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase (run every val_freq epochs to save time)
        avg_val_loss = 0.0
        avg_dice_scores = np.zeros((num_classes,))
        if epoch % val_freq == 0:
            model.eval()
            val_loss = 0.0
            all_dice_scores = []

            with torch.no_grad():
                val_pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                                desc=f"Epoch {epoch+1}/{num_epochs} - Val", leave=False)
                for batch_idx, (images, labels) in val_pbar:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    if use_amp:
                        with autocast(device_type=amp_device):
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item()

                    # Calculate Dice coefficient
                    dice_scores = dice_coefficient(outputs, labels, num_classes=num_classes)
                    all_dice_scores.append(dice_scores)

                    mean_batch_dice = np.mean(dice_scores[1:]) if num_classes > 1 else dice_scores[0]
                    val_pbar.set_postfix(val_loss=f"{loss.item():.4f}", mean_dice=f"{mean_batch_dice:.4f}")

            avg_val_loss = val_loss / len(val_loader)
            avg_dice_scores = np.mean(all_dice_scores, axis=0)
            mean_dice = np.mean(avg_dice_scores[1:])  # Exclude background (class 0)
        else:
            # Skip validation this epoch: keep previous metrics
            mean_dice = best_dice
            avg_dice_scores = np.zeros((num_classes,))
        
        # 在验证后更新权重
        if epoch % val_freq == 0:
            # 更新历史记录
            weight_adjuster.update(avg_dice_scores)
            
            # 调整权重
            if epoch >= weight_adjuster.window_size:
                new_weights = weight_adjuster.adjust_weights()
                criterion.class_weights = new_weights.to(device)
                print("\n权重更新:")
                for c in range(num_classes):
                    print(f"类别 {c}: {new_weights[c]:.4f}")
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        if epoch % val_freq == 0:
            # 只有在真正运行了验证时才打印Val Loss
            print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        else:
            # 跳过验证的epoch只打印训练损失
            print(f"  Train Loss: {avg_train_loss:.4f}")
            
        if np.any(avg_dice_scores):
            print(f"  Dice per class: {[f'{d:.4f}' for d in avg_dice_scores]}")
            print(f"  Mean Dice (excl. bg): {mean_dice:.4f}")
        else:
            print(f"  Validation skipped this epoch (val_freq={val_freq}).")

        # 只在实际运行验证时调用 scheduler
        if epoch % val_freq == 0:
            scheduler.step(mean_dice)
        
        # Save best model
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'dice_per_class': avg_dice_scores,
            }, 'best_model.pth')
            print(f"  ✓ Model saved with mean Dice: {best_dice:.4f}")
        
        # Early stopping if Dice > 0.7 for all classes (if validated)
        if np.any(avg_dice_scores) and np.all(avg_dice_scores[1:] >= 0.7):
            print(f"\n🎉 Target achieved! All classes have Dice ≥ 0.7")
            break

# --- Dynamic Weight Adjuster ---
class DynamicWeightAdjuster:
    def __init__(self, initial_weights, window_size=3, 
                 target_dice=0.7, min_improvement=0.01):
        """
        通用的动态权重调整器
        Args:
            initial_weights: 初始权重
            window_size: 观察窗口大小
            target_dice: 目标 Dice 分数
            min_improvement: 最小期望改善
        """
        self.weights = initial_weights.clone()
        self.history = {i: [] for i in range(len(initial_weights))}
        self.window_size = window_size
        self.target_dice = target_dice
        self.min_improvement = min_improvement
        
    def update(self, dice_scores):
        """更新每个类别的 Dice 分数历史"""
        for i, dice in enumerate(dice_scores):
            self.history[i].append(dice)
            if len(self.history[i]) > self.window_size:
                self.history[i].pop(0)
    
    def adjust_weights(self):
        """根据性能动态调整权重"""
        for i in range(len(self.weights)):
            if len(self.history[i]) >= self.window_size:
                # 计算最近几个 epoch 的改善程度
                recent_scores = self.history[i][-self.window_size:]
                improvements = [recent_scores[j+1] - recent_scores[j] 
                              for j in range(len(recent_scores)-1)]
                avg_improvement = sum(improvements) / len(improvements)
                current_dice = recent_scores[-1]
                
                # 根据当前性能和改善程度调整权重
                if current_dice < self.target_dice:
                    # 性能越差，增幅越大
                    gap = self.target_dice - current_dice
                    if avg_improvement < self.min_improvement:
                        increase_factor = 1.0 + gap
                        self.weights[i] *= increase_factor
                        print(f"类别 {i} - Dice: {current_dice:.4f}, "
                              f"增加权重: {increase_factor:.2f}x")
                elif current_dice > self.target_dice + 0.1:
                    # 性能明显超过目标，适当降低权重
                    self.weights[i] *= 0.9
                    print(f"类别 {i} - Dice: {current_dice:.4f}, 降低权重")
        
        # 归一化权重
        self.weights = self.weights / self.weights.mean()
        # 限制权重范围，避免过大或过小
        self.weights = torch.clamp(self.weights, min=0.1, max=5.0)
        
        return self.weights

def plot_class_weights(class_weights, save_path="class_weights.png"):
    """
    绘制类别权重图并保存为图片
    Args:
        class_weights (torch.Tensor): 类别权重
        save_path (str): 保存路径
    """
    num_classes = len(class_weights)
    plt.figure(figsize=(8, 6))
    plt.bar(range(num_classes), class_weights.cpu().numpy(), color='skyblue')
    plt.xlabel("Class Index")
    plt.ylabel("Weight")
    plt.title("Class Weights")
    plt.xticks(range(num_classes))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Class weights plot saved to {save_path}")

def plot_class_distribution(class_counts, save_path="class_distribution.png"):
    """
    绘制类别分布图并在柱状图上显示具体的体素数量
    Args:
        class_counts (torch.Tensor): 每个类别的体素数量
        save_path (str): 保存路径
    """
    num_classes = len(class_counts)
    counts = class_counts.cpu().numpy()  # 转为 NumPy 数组

    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(num_classes), counts, color='skyblue')

    # 在柱状图顶部显示具体的体素数量
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X 坐标
            bar.get_height(),                  # Y 坐标（柱子高度）
            f"{int(count):,}",                 # 显示的文本（体素数量，带千分位）
            ha='center',                       # 水平居中
            va='bottom',                       # 垂直方向在柱子顶部
            fontsize=10,                       # 字体大小
            color='black'                      # 字体颜色
        )

    plt.xlabel("Class Index")
    plt.ylabel("Voxel Count")
    plt.title("Class Distribution")
    plt.xticks(range(num_classes))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Class distribution plot saved to {save_path}")

# --- Main ---
if __name__ == "__main__":
    # 固定随机性（可重现） - 用于可重复性。若优先速度可禁用下面两行以允许 cudnn benchmark
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For best speed on fixed-size inputs enable benchmark and disable strict determinism
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = r"C:\Users\17561\Desktop\new 3710\data"
    
    # 添加缺失的配置参数
    CONFIG = {
        'num_classes': 6,
        'target_size': (128, 128, 64),
        'batch_size': 2,
        'num_workers': 4,
        'base_filters': 16,
        'val_freq': 1,  # 每个epoch都验证
        'grad_accum_steps': 1,  # 梯度累积步数
        'use_preload': False,
    }
    
    print(f"Using device: {device}\n")
    
    # 创建数据集（只创建一次）
    train_dataset = Prostate3DDataset(
        data_root=data_root,
        split='train',
        num_classes=CONFIG['num_classes'],
        target_size=CONFIG['target_size'],
        preload=CONFIG['use_preload']
    )
    
    val_dataset = Prostate3DDataset(
        data_root=data_root,
        split='val',
        num_classes=CONFIG['num_classes'],
        target_size=CONFIG['target_size'],
        preload=CONFIG['use_preload']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
    )
    
    val_loader = DataLoader(

        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, CONFIG['num_workers']//2),
        pin_memory=True
    )
    
    # 初始化模型
    model = UNet3D(
        in_channels=1, 
        num_classes=CONFIG['num_classes'], 
        base_filters=CONFIG['base_filters']
    ).to(device)
    model.apply(init_weights_he)
    
    print("Model initialized from scratch.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 计算类别权重（只计算训练集的权重）
    class_weights = train_dataset.class_weights
    print("\n=== 类别权重分析 ===")
    print(f"类别权重: {[f'{w:.4f}' for w in class_weights]}")

    # 绘制类别分布图
    plot_class_distribution(train_dataset.class_counts, save_path="class_distribution.png")

    # 绘制权重图
    plot_class_weights(class_weights, save_path="class_weights.png")
    
    # 开始训练
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        device=device,
        num_classes=CONFIG['num_classes'],
        val_freq=CONFIG['val_freq'],
        grad_accum_steps=CONFIG['grad_accum_steps']
    )