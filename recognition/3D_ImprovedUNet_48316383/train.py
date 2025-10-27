import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm  
import torch.nn as nn
import torch.optim as optim
from modules import UNet3D  
from dataset import Prostate3DDataset, MONAIAugmentation, create_monai_dataloaders
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler  
from config import CONFIG  
from utils import init_weights_he, dice_coefficient, plot_class_weights, plot_class_distribution
from training_visualizer import TrainingVisualizer

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
def validate_model(model, val_loader, criterion, device, use_amp=True):
    """Validate the model and return validation loss and dice scores"""
    model.eval()
    val_loss = 0.0
    all_dice_scores = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Handle both MONAI dict format and tuple format
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                targets = batch['label'].to(device)
            else:
                images, targets = batch
                images, targets = images.to(device), targets.to(device)
            
            # Use mixed precision for validation if enabled
            if use_amp:
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
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
def train_model(model, train_loader, val_loader, optimizer, scheduler, weight_adjuster, device, num_epochs, val_freq,
                use_early_stopping=False, patience=10, min_delta=0.01, num_worst_classes_to_track=2):
    """
    Train the model with dynamic tracking of the worst-performing classes.
    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        weight_adjuster: Dynamic weight adjuster for class weights.
        device: Device to use for training (e.g., 'cuda' or 'cpu').
        num_epochs: Total number of epochs to train.
        val_freq: Frequency of validation (in epochs).
        use_early_stopping: Whether to use early stopping.
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum improvement required to reset the early stopping counter.
        num_worst_classes_to_track: Number of worst-performing classes to track.
    """
    # Initialize loss function (using CONFIG parameters)
    criterion = FocalDiceLoss(
        alpha=CONFIG.get('loss_alpha', 1.0),
        gamma=CONFIG.get('loss_gamma', 1.0),
        dice_weight=CONFIG.get('loss_dice_weight', 1.0),
        focal_weight=CONFIG.get('loss_focal_weight', 20.0),
        smooth=CONFIG.get('loss_smooth', 1.0),
        class_weights=weight_adjuster.weights.to(device)
    )

    # Initialize EarlyStopper and tracking variables
    if use_early_stopping:
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    best_min_dice_worst_k = -1.0  # Track the best minimum Dice score for the worst K classes
    best_model_state = None  # Store best model state for saving later
    best_epoch = 0  # Track which epoch had the best performance

    # Initialize TrainingVisualizer if enabled
    visualizer = None
    if CONFIG.get('enable_visualization', False):
        plot_dir = CONFIG.get('plot_dir', './training_plots')
        visualizer = TrainingVisualizer(
            num_classes=CONFIG['num_classes'],
            save_dir=plot_dir,
            metrics_to_track=CONFIG.get('metrics_to_plot', None)
        )
        print(f"📊 Training visualization enabled! Plots will be saved to: {plot_dir}")

    # Initialize GradScaler for mixed precision training (if enabled)
    use_amp = CONFIG.get('use_amp', True)
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("✨ Mixed Precision Training (AMP) enabled!")
    else:
        print("📝 Training without AMP (full precision)")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        dice_scores_per_epoch = []

        # Training loop
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            # Handle both MONAI dict format and tuple format
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                targets = batch['label'].to(device)
            else:
                images, targets = batch
                images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            
            # Conditional mixed precision forward pass
            if use_amp:
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

            # Calculate Dice scores for the current batch
            dice_scores = dice_coefficient(outputs, targets, targets.shape[1])
            mean_batch_dice = sum(dice_scores) / len(dice_scores)
            dice_scores_per_epoch.append(dice_scores)

            # Update progress bar
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", mean_dice=f"{mean_batch_dice:.4f}")

        # Calculate average Dice score for the epoch
        mean_epoch_dice = sum([sum(scores) / len(scores) for scores in dice_scores_per_epoch]) / len(dice_scores_per_epoch)

        print(f"\n[Training] Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss = {train_loss:.4f}, Mean Train Dice = {mean_epoch_dice:.4f}")
        print("  Per-Class Train Dice Scores:")
        for c in range(targets.shape[1]):
            class_dice = sum(scores[c] for scores in dice_scores_per_epoch) / len(dice_scores_per_epoch)
            print(f"    Class {c}: {class_dice:.4f}")

        # Validation and weight adjustment
        if epoch % val_freq == 0:
            model.eval()
            val_loss, avg_dice_scores = validate_model(model, val_loader, criterion, device, use_amp=use_amp)

            # Identify the worst-performing K classes
            dice_non_bg_with_indices = list(enumerate(avg_dice_scores[1:], start=1))  # [(1, dice1), (2, dice2), ...]
            dice_non_bg_with_indices.sort(key=lambda x: x[1])  # Sort by Dice score (ascending)
            worst_k_dice_scores = [score for index, score in dice_non_bg_with_indices[:num_worst_classes_to_track]]
            worst_k_indices = [index for index, score in dice_non_bg_with_indices[:num_worst_classes_to_track]]

            # Calculate the minimum Dice score among the worst K classes
            min_dice_worst_k = np.min(worst_k_dice_scores) if worst_k_dice_scores else -1.0

            # Update weights dynamically
            weight_adjuster.update(avg_dice_scores)
            new_weights = weight_adjuster.adjust_weights()
            criterion.class_weights = new_weights.to(device)

            # Print validation results
            mean_val_dice_non_bg = np.mean(avg_dice_scores[1:])  # Mean Dice excluding background
            print(f"\n[Validation] Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Val Loss = {val_loss:.4f}, Mean Val Dice (non-bg) = {mean_val_dice_non_bg:.4f}")
            print(f"  Per-Class Val Dice Scores:")
            for c, dice in enumerate(avg_dice_scores):
                print(f"    Class {c}: {dice:.4f}")
            print(f"  Worst {num_worst_classes_to_track} classes: {worst_k_indices}")
            print(f"  Min Dice (Worst {num_worst_classes_to_track}): {min_dice_worst_k:.4f}")
            print(f"  Updated Weights: {[f'{w:.4f}' for w in new_weights]}")
            
            # Update TrainingVisualizer with metrics
            if visualizer is not None:
                metrics = {
                    'train_loss': train_loss / len(train_loader),
                    'val_loss': val_loss,
                    'mean_dice_train': mean_epoch_dice,
                    'mean_dice_val': mean_val_dice_non_bg,
                    'class_dice_val': avg_dice_scores,
                    'worst_k_dice': min_dice_worst_k,
                    'lr': optimizer.param_groups[0]['lr']
                }
                visualizer.update(epoch + 1, metrics)

            # Track the best model (don't save yet, only record the state)
            if min_dice_worst_k > best_min_dice_worst_k:
                best_min_dice_worst_k = min_dice_worst_k
                best_epoch = epoch + 1
                # Store the best model state in memory
                best_model_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict().copy(),
                    'optimizer_state_dict': optimizer.state_dict().copy(),
                    'best_dice': best_min_dice_worst_k,
                    'config': CONFIG
                }
                print(f"  🎉 New BEST model found at epoch {epoch + 1}! MIN Dice (Worst {num_worst_classes_to_track}): {best_min_dice_worst_k:.4f}")

            # Early stopping based on the worst K classes' minimum Dice score
            if use_early_stopping and early_stopper.early_stop(min_dice_worst_k):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                break

        # Step the learning rate scheduler
        scheduler.step()

    # Get actual training info
    actual_epoch_stopped = epoch + 1  # Actual epoch where training stopped
    configured_epochs = num_epochs  # Configured/planned total epochs
    
    print("\nTraining finished!")
    print(f"   Configured epochs: {configured_epochs}")
    print(f"   Actual epochs trained: {actual_epoch_stopped}")
    
    # Save the best model (only once at the end of training or after early stopping)
    # Use configured epochs in filename, not actual stopped epoch
    if best_model_state is not None:
        best_model_path = f"best_model_{configured_epochs}epochs.pth"
        torch.save(best_model_state, best_model_path)
        print(f"\n💾 Best model saved to: {best_model_path}")
        print(f"   Best performance at epoch: {best_epoch}/{configured_epochs}")
        print(f"   Best MIN Dice (Worst {num_worst_classes_to_track}): {best_min_dice_worst_k:.4f}")
    else:
        print("\n⚠️  No best model recorded (validation may not have run)")
    
    # Generate final training visualization and report with configured epochs suffix
    if visualizer is not None:
        print("\n" + "="*80)
        print("Generating final training summary...")
        print("="*80)
        
        # Use configured epochs in filename for consistency
        visualizer.plot_final_summary(epoch_suffix=f"{configured_epochs}epochs")
        visualizer.generate_final_report(epoch_suffix=f"{configured_epochs}epochs")
        
        print("✅ Training visualization complete!")

# --- Early Stopper ---
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.01):
        """
        Early stopping to stop training when performance stops improving.
        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        
    def early_stop(self, score):
        """
        Check if training should stop.
        Args:
            score: Current validation score (higher is better).
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            return False
            
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

# --- Dynamic Weight Adjuster ---
class DynamicWeightAdjuster:
    def __init__(self, initial_weights, window_size=1, target_dice=0.7, min_improvement=0.01):
        """
        Dynamic adjuster for class weights based on recent performance.
        Args:
            initial_weights: Initial weights.
            window_size: Size of the observation window.
            target_dice: Target Dice score.
            min_improvement: Minimum expected improvement.
        """
        self.weights = initial_weights.clone()
        self.history = {i: [] for i in range(len(initial_weights))}
        self.window_size = window_size
        self.target_dice = target_dice
        self.min_improvement = min_improvement

    def update(self, dice_scores):
        """Update the Dice score history for each class."""
        for i, dice in enumerate(dice_scores):
            self.history[i].append(dice)
            if len(self.history[i]) > self.window_size:
                self.history[i].pop(0)

    def adjust_weights(self):
        """Adjust weights dynamically based on performance."""
        for i in range(len(self.weights)):
            if len(self.history[i]) >= self.window_size:
                # Calculate the improvement over the last few epochs
                recent_scores = self.history[i][-self.window_size:]
                if len(recent_scores) > 1:
                    avg_improvement = (recent_scores[-1] - recent_scores[0]) / (len(recent_scores) - 1)
                else:
                    avg_improvement = 0  # 如果只有一个分数，没有改进
                current_dice = recent_scores[-1]

                # Adjust weights based on current performance and improvement
                if current_dice < self.target_dice:
                    gap = self.target_dice - current_dice
                    if avg_improvement < self.min_improvement:
                        self.weights[i] *= (1.0 + gap)  # Increase weight for underperforming classes
                elif current_dice > self.target_dice + 0.1:
                    self.weights[i] *= 0.9  # Decrease weight for overperforming classes

                # Debugging: Print weight adjustment details
                print(f"Class {i}: Current Dice = {current_dice:.4f}, Avg Improvement = {avg_improvement:.4f}, New Weight = {self.weights[i]:.4f}")

        # Normalize and clamp weights
        self.weights = self.weights / self.weights.mean()
        self.weights = torch.clamp(self.weights, min=0.1, max=15.0)
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
    # Set random seed for reproducibility (read from CONFIG)
    # If speed is the priority, you can set seed to None in CONFIG to disable deterministic behavior
    seed = CONFIG.get('seed', 42)
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

    # use CONFIG parameters
    print(f"Using device: {device}")
    print(f"Mixed Precision (AMP): {CONFIG.get('use_amp', True)}")
    print(f"Batch Size: {CONFIG.get('train_batch_size', CONFIG.get('batch_size', 2))}")
    print(f"Base Filters: {CONFIG['base_filters']}")
    print(f"Visualization: {CONFIG.get('enable_visualization', False)}\n")

    # DATA LOADING
    use_monai_loader = CONFIG.get('use_monai_loader', False)
    
    if use_monai_loader:
        print("📦 Using MONAI-based data loading...")
        train_loader, val_loader, _ = create_monai_dataloaders(
            CONFIG, 
            use_cache=CONFIG.get('cache_rate', 0) > 0
        )
        
        # Compute class weights from MONAI loader
        from dataset import compute_class_weights_monai
        class_weights = compute_class_weights_monai(
            train_loader, 
            CONFIG['num_classes'], 
            device=device
        )
        
    else:
        print("📦 Using custom Prostate3DDataset...")
        # Create data augmentation instance from CONFIG (MONAI-based)
        aug_config = CONFIG.get('augmentation', {})
        augmentation = MONAIAugmentation(
            flip_prob=aug_config.get('flip_prob', 0.5),
            noise_std=aug_config.get('noise_std', 0.05)
        )

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
            batch_size=CONFIG.get('train_batch_size', CONFIG.get('batch_size', 2)),
            shuffle=True,
            num_workers=CONFIG['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG.get('val_batch_size', 1),
            shuffle=False,
            num_workers=max(1, CONFIG['num_workers'] // 2),
            pin_memory=True
        )
        
        # Compute class weights from custom dataset
        class_weights = train_dataset.class_weights
        print("\n=== Class Weight Analysis ===")
        print(f"Class Weights: {[f'{w:.4f}' for w in class_weights]}")

        # Plot class distribution
        plot_class_distribution(train_dataset.class_counts, save_path="class_distribution.png")
    
    # Initialize model using CONFIG
    model = UNet3D(
        in_channels=CONFIG.get('in_channels', 1),
        num_classes=CONFIG['num_classes'],
        base_filters=CONFIG['base_filters']
    ).to(device)
    model.apply(init_weights_he)
    
    print("\n" + "="*80)
    print("Model initialized from scratch.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("="*80 + "\n")

    # Plot weight map
    plot_class_weights(class_weights, save_path="class_weights.png")

    # Define optimizer and learning rate scheduler (using CONFIG parameters)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=CONFIG.get('learning_rate', 1e-3), 
        weight_decay=CONFIG.get('weight_decay', 1e-5)
    )
    
    # Exponential learning rate decay
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, 
        gamma=CONFIG.get('lr_decay_gamma', 0.985)
    )

    # Instantiate DynamicWeightAdjuster (using CONFIG parameters)
    weight_adjuster = DynamicWeightAdjuster(
        initial_weights=class_weights,  # Use class_weights (works for both MONAI and custom loader)
        window_size=CONFIG.get('weight_window_size', 3),
        target_dice=CONFIG.get('target_dice', 0.7),
        min_improvement=CONFIG.get('min_improvement', 0.01)
    )

    # Start training (all parameters from CONFIG)
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        weight_adjuster=weight_adjuster,  
        num_epochs=CONFIG['num_epochs'],  # Read from CONFIG
        device=device,
        val_freq=CONFIG['val_freq'],
        use_early_stopping=CONFIG['use_early_stopping'],
        patience=CONFIG['patience'],
        min_delta=CONFIG['min_delta'],
        num_worst_classes_to_track=CONFIG.get('num_worst_classes_to_track', 2)
    )