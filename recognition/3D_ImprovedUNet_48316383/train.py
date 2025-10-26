import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm  
import torch.nn as nn
import torch.optim as optim
from modules import UNet3D  
from dataset import Prostate3DDataset, RandomAugmentation
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler  
from config import CONFIG  
from utils import init_weights_he, dice_coefficient, plot_class_weights, plot_class_distribution

# --- Weighted Dice Loss for imbalanced classes ---
class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, class_weights=None, weight_power=1.0):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.class_weights = class_weights
        self.weight_power = weight_power  # control the strength of weights
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)

        # Calculate Dice for each class
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)
        
        if self.class_weights is not None:
            weights = self.class_weights.to(pred.device)
            # Use weight_power to adjust the influence of weights
            weights = weights ** self.weight_power
            dice_per_class = dice_per_class * weights.unsqueeze(0)

        # Print Dice values for each class for monitoring
        if not self.training:  # Print during validation
            for c in range(len(dice_per_class[0])):
                print(f"Class {c} Dice: {dice_per_class[0][c]:.4f}")
        
        return 1 - dice_per_class.mean()

# --- Focal Dice Loss ---
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, dice_weight=1.0, focal_weight=20.0, smooth=1.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, pred, target):
        # suppose target already One-Hot encoded
        target_one_hot = target.float()

        # calculate Dice Loss
        pred_softmax = torch.softmax(pred, dim=1)
        intersection = (pred_softmax * target_one_hot).sum(dim=(2, 3, 4))
        union = pred_softmax.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        if self.class_weights is not None:
            dice = dice * self.class_weights.to(dice.device)

        dice_loss = 1 - dice.mean()

        # calculate Focal Loss
        pt = (pred_softmax * target_one_hot).sum(dim=1)  # (N, D, H, W)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = -torch.log(pt + 1e-8) * focal_weight
        focal_loss = focal_loss.mean()

        # Combine losses
        total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        return total_loss




# --- Validation Function ---
def validate_model(model, val_loader, criterion, device):
    """Validate the model and return validation loss and dice scores"""
    model.eval()
    val_loss = 0.0
    all_dice_scores = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            # Calculate dice scores for this batch
            dice_scores = dice_coefficient(outputs, targets, targets.shape[1])
            all_dice_scores.append(dice_scores)

    # Average the validation loss
    val_loss /= len(val_loader)
    
    # Average dice scores across all batches
    avg_dice_scores = [sum(scores[i] for scores in all_dice_scores) / len(all_dice_scores) 
                       for i in range(len(all_dice_scores[0]))]
    
    return val_loss, avg_dice_scores

# --- Training Function (AMP, gradient accumulation, optional compile) ---
def train_model(model, train_loader, val_loader, optimizer, scheduler, weight_adjuster, device, num_epochs, val_freq):
    # initialize FocalDiceLoss
    criterion = FocalDiceLoss(
        alpha=1.0,
        gamma=1.0,  
        dice_weight=1.0,
        focal_weight=20.0,  
        smooth=1.0,
        class_weights=weight_adjuster.weights.to(device)  # use DynamicWeightAdjuster initial weights.to(device
    )

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        dice_scores_per_epoch = []

        # use tqdm to wrap the training loop
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            images, targets = batch
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # calculate Dice scores for this batch
            dice_scores = dice_coefficient(outputs, targets, targets.shape[1])
            mean_batch_dice = sum(dice_scores) / len(dice_scores)
            dice_scores_per_epoch.append(dice_scores)

            # update progress bar display
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", mean_dice=f"{mean_batch_dice:.4f}")

        # calculate the average Dice score for the current epoch
        mean_epoch_dice = sum([sum(scores) / len(scores) for scores in dice_scores_per_epoch]) / len(dice_scores_per_epoch)

        # print Dice scores for the epoch
        print(f"\n[Training] Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss = {train_loss:.4f}, Mean Train Dice = {mean_epoch_dice:.4f}")
        print("  Per-Class Train Dice Scores:")
        for c in range(targets.shape[1]):
            class_dice = sum(scores[c] for scores in dice_scores_per_epoch) / len(dice_scores_per_epoch)
            print(f"    Class {c}: {class_dice:.4f}")

        # Validate and adjust weights
        if epoch % val_freq == 0:
            model.eval()
            val_loss, avg_dice_scores = validate_model(model, val_loader, criterion, device)

            # Adjust weights dynamically
            new_weights = weight_adjuster.adjust_weights()  # use the passed instance to adjust weights
            criterion.class_weights = new_weights.to(device)  # update the loss function's weights

            print(f"\n[Validation] Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Val Loss = {val_loss:.4f}, Mean Val Dice = {sum(avg_dice_scores) / len(avg_dice_scores):.4f}")
            print("  Per-Class Val Dice Scores:")
            for c, dice in enumerate(avg_dice_scores):
                print(f"    Class {c}: {dice:.4f}")
            print(f"  Updated Weights: {[f'{w:.4f}' for w in new_weights]}")

# --- Dynamic Weight Adjuster ---
class DynamicWeightAdjuster:
    def __init__(self, initial_weights, window_size=3, target_dice=0.7, min_improvement=0.01):
        """
        dynamic adjuster for class weights based on recent performance
        Args:
            initial_weights: initial weights
            window_size: size of the observation window
            target_dice: target Dice score
            min_improvement: minimum expected improvement
        """
        self.weights = initial_weights.clone()
        self.history = {i: [] for i in range(len(initial_weights))}
        self.window_size = window_size
        self.target_dice = target_dice
        self.min_improvement = min_improvement

    def update(self, dice_scores):
        """Update the Dice score history for each class"""
        for i, dice in enumerate(dice_scores):
            self.history[i].append(dice)
            if len(self.history[i]) > self.window_size:
                self.history[i].pop(0)

    def adjust_weights(self):
        """Adjust weights dynamically based on performance"""
        for i in range(len(self.weights)):
            if len(self.history[i]) >= self.window_size:
                # Calculate the improvement over the last few epochs
                recent_scores = self.history[i][-self.window_size:]
                avg_improvement = sum(recent_scores[j+1] - recent_scores[j] for j in range(len(recent_scores)-1)) / len(recent_scores)
                current_dice = recent_scores[-1]

                # Adjust weights based on current performance and improvement
                if current_dice < self.target_dice:
                    # The worse the performance, the greater the increase
                    gap = self.target_dice - current_dice
                    if avg_improvement < self.min_improvement:
                        self.weights[i] *= (1.0 + gap)
                elif current_dice > self.target_dice + 0.1:
                    # If performance significantly exceeds the target, reduce the weight appropriately
                    self.weights[i] *= 0.9

        # Limit weight range
        self.weights = self.weights / self.weights.mean()
        self.weights = torch.clamp(self.weights, min=0.1, max=15.0)  # Raise weight upper limit to 15.0
        
        return self.weights

def plot_class_weights(class_weights, save_path="class_weights.png"):
    """
    Plot class weights and save as image
    Args:
        class_weights (torch.Tensor): Class weights
        save_path (str): Save path
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
    Plot class distribution and display voxel counts on the bar chart
    Args:
        class_counts (torch.Tensor): Voxel counts for each class
        save_path (str): Save path
    """
    num_classes = len(class_counts)
    counts = class_counts.cpu().numpy()  # Convert to NumPy array

    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(num_classes), counts, color='skyblue')

    # Display voxel counts on top of the bars
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X coordinate
            bar.get_height(),                  # Y coordinate (bar height)
            f"{int(count):,}",                 # Text to display (voxel count, with thousands separator)
            ha='center',                       # Horizontal alignment
            va='bottom',                       # Vertical alignment (at the top of the bar)
            fontsize=10,                       # Font size
            color='black'                      # Font color
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
    #  Fixed randomness (reproducible) - for reproducibility. 
    # If speed is the priority, you can disable the following two lines to allow cudnn benchmark
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

    # use parameters from CONFIG
    print(f"Using device: {device}\n")

    # Create data augmentation instance
    augmentation = RandomAugmentation(flip_prob=0.5, noise_std=0.05, rotate_angle=0)

    # Create training dataset with data augmentation
    train_dataset = Prostate3DDataset(
        data_root=CONFIG['data_root'],
        split='train',
        num_classes=CONFIG['num_classes'],
        target_size=CONFIG['target_size'],
        preload=CONFIG['use_preload'],
        transform=augmentation  # use data augmentation
    )

    # Validation dataset does not require data augmentation
    val_dataset = Prostate3DDataset(
        data_root=CONFIG['data_root'],
        split='val',
        num_classes=CONFIG['num_classes'],
        target_size=CONFIG['target_size'],
        preload=CONFIG['use_preload']
    )

    # Create data loader
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
        num_workers=max(1, CONFIG['num_workers'] // 2),
        pin_memory=True
    )
    
    # Initialize model
    model = UNet3D(
        in_channels=1, 
        num_classes=CONFIG['num_classes'], 
        base_filters=CONFIG['base_filters']
    ).to(device)
    model.apply(init_weights_he)
    
    print("Model initialized from scratch.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compute class weights (only for training set)
    class_weights = train_dataset.class_weights
    print("\n=== Class Weight Analysis ===")
    print(f"Class Weights: {[f'{w:.4f}' for w in class_weights]}")

    # Plot class distribution
    plot_class_distribution(train_dataset.class_counts, save_path="class_distribution.png")

    # Plot weight map
    plot_class_weights(class_weights, save_path="class_weights.png")

    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)  # use ExponentialLR

    # Instantiate DynamicWeightAdjuster
    weight_adjuster = DynamicWeightAdjuster(
        initial_weights=train_dataset.class_weights,
        window_size=3,
        target_dice=0.7,
        min_improvement=0.01
    )

    # Start training
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        weight_adjuster=weight_adjuster,  # pass the instantiated DynamicWeightAdjuster
        num_epochs=100,
        device=device,
        val_freq=CONFIG['val_freq']
    )