import torch
import numpy as np
import matplotlib.pyplot as plt

# Initialize weights using He initialization
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

# Calculate Dice coefficient for each class
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

# Plot class weights
def plot_class_weights(class_weights, save_path="class_weights.png"):
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

# Plot class distribution
def plot_class_distribution(class_counts, save_path="class_distribution.png"):
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

# Convert labels to One-Hot encoding
def to_channels(arr: np.ndarray, num_classes=6, dtype=np.uint8) -> np.ndarray:
    """Convert label array to One-Hot encoding"""
    res = np.zeros((num_classes,) + arr.shape, dtype=dtype)  # (C, D, H, W)
    for c in range(num_classes):
        res[c] = (arr == c).astype(dtype)
    return res