import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from modules import UNet3D
from dataset import Prostate3DDataset

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
    def __init__(self, smooth=1.0, class_weights=None):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.class_weights = class_weights
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        if self.class_weights is not None:
            weights = self.class_weights.to(pred.device)
            dice = dice * weights.unsqueeze(0)
        
        return 1 - dice.mean()

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
                val_freq=1, grad_accum_steps=1):  # 改为 val_freq=1
    # 调整 class weights 使其更温和
    # 原始频率: [49.97%, 44.94%, 3.78%, 0.79%, 0.28%, 0.24%]
    # 使用平方根倒数而非直接倒数，避免权重过大
    class_weights = torch.FloatTensor([1.0, 1.05, 3.5, 8.0, 13.0, 15.0])
    class_weights = class_weights / class_weights.sum() * num_classes
    
    criterion = WeightedDiceLoss(class_weights=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # 提高初始学习率
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
    print(f"Using device: {device}\n")
    
    # Dataset root
    data_root = r"C:\Users\17561\Desktop\new 3710\data"
    num_classes = 6
    
    # 选择加载方式
    USE_PRELOAD = False           # 按需加载，节省内存
    target_size = (128, 128, 64)    # adjust if needed
    batch_size = 2                  # if GPU allows; else reduce to 1
    num_workers = 4
    base_filters = 32               # try increasing if memory allows
    
    # Performance-oriented DataLoader flags
    dl_kwargs = dict(pin_memory=True, persistent_workers=True, prefetch_factor=2) if num_workers>0 else dict(pin_memory=True)
    
    # Create datasets
    if USE_PRELOAD:
        print("Using PRELOAD mode (load_data_3D)")
        train_dataset = Prostate3DDataset(data_root, split='train', num_classes=num_classes,
                                         target_size=target_size, preload=True)
        val_dataset = Prostate3DDataset(data_root, split='val', num_classes=num_classes,
                                       target_size=target_size, preload=True)
    else:
        print("Using LAZY LOAD mode (on-demand)")
        train_dataset = Prostate3DDataset(data_root, split='train', num_classes=num_classes,
                                         target_size=target_size, preload=False)
        val_dataset = Prostate3DDataset(data_root, split='val', num_classes=num_classes,
                                       target_size=target_size, preload=False)
    
    # Create dataloaders (use smaller batch for val to reduce mem pressure)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, **dl_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                           num_workers=max(1, num_workers//2), pin_memory=True)
    
    # Initialize model
    model = UNet3D(in_channels=1, num_classes=num_classes, base_filters=base_filters).to(device)

    # Try to compile model only if Triton is available (avoid runtime failure)
    triton_ok = False
    try:
        import triton  # check if triton exists
        triton_ok = True
    except (ModuleNotFoundError, ImportError):
        triton_ok = False

    if triton_ok:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile()")
        except Exception as e:
            print("torch.compile() failed, continuing without compile. Error:", e)
    else:
        print("Triton not available — skipping torch.compile(). Install triton if you want compilation.")

    # 显式初始化权重（He/Kaiming）
    model.apply(init_weights_he)

    print("Model initialized from scratch. No pretrained weights loaded.")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Train with acceleration options
    VAL_FREQ = 1  # 改为每轮都验证，便于调试
    GRAD_ACCUM_STEPS = 1
    
    # 先用较小的模型和尺寸测试
    target_size = (96, 96, 48)
    base_filters = 24
    batch_size = 2
    
    # --- Data Sanity Check ---
    print("\n=== Data Sanity Check ===")
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Image shape: {images.shape}, range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Label shape: {labels.shape}")
        # 检查每个类别的体素数
        for c in range(num_classes):
            count = (labels[:, c] > 0.5).sum().item()
            print(f"  Class {c}: {count} voxels")
        if batch_idx >= 2:  # 只检查前3个batch
            break
    print("=========================\n")

    train_model(model, train_loader, val_loader, num_epochs=100, device=device, num_classes=num_classes,
                val_freq=VAL_FREQ, grad_accum_steps=GRAD_ACCUM_STEPS)