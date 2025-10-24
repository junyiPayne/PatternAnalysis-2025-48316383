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

# --- Training Function ---
def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda', num_classes=6):
    # Class weights based on inverse frequency (from check_data output)
    # Class distribution: [49.97%, 44.94%, 3.78%, 0.79%, 0.28%, 0.24%]
    class_weights = torch.FloatTensor([1.0, 1.1, 13.2, 63.3, 178.5, 208.3])
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize
    
    criterion = WeightedDiceLoss(class_weights=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_dice_scores = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate Dice coefficient
                dice_scores = dice_coefficient(outputs, labels, num_classes=num_classes)
                all_dice_scores.append(dice_scores)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_dice_scores = np.mean(all_dice_scores, axis=0)
        mean_dice = np.mean(avg_dice_scores[1:])  # Exclude background (class 0)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"  Dice per class: {[f'{d:.4f}' for d in avg_dice_scores]}")
        print(f"  Mean Dice (excl. bg): {mean_dice:.4f}")
        
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
        
        # Early stopping if Dice > 0.7 for all classes
        if np.all(avg_dice_scores[1:] >= 0.7):
            print(f"\n🎉 Target achieved! All classes have Dice ≥ 0.7")
            break

# --- Main ---
if __name__ == "__main__":
    # 固定随机性（可重现）
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Dataset root
    data_root = r"C:\Users\17561\Desktop\new 3710\data"
    num_classes = 6
    
    # 选择加载方式
    USE_PRELOAD = False           # 按需加载，节省内存
    target_size = (96, 96, 48)    # 更小的尺寸
    batch_size = 1
    num_workers = 2
    base_filters = 16             # UNet3D 的基础通道数
    
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
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    # Initialize model
    model = UNet3D(in_channels=1, num_classes=num_classes, base_filters=base_filters).to(device)
    
    # 显式初始化权重（He/Kaiming）
    model.apply(init_weights_he)

    print("Model initialized from scratch. No pretrained weights loaded.")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Train
    train_model(model, train_loader, val_loader, num_epochs=100, device=device, num_classes=num_classes)