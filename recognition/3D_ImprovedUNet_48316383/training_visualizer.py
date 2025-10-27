"""
Training Visualization Module
==============================

Real-time tracking and plotting of training metrics including:
- Training and validation loss
- Dice scores (overall and per-class)
- Learning rate
- Worst-K class performance

Author: Pattern Analysis Team
Date: 2025-10-27
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server compatibility
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class TrainingVisualizer:
    """
    Tracks and visualizes training metrics in real-time
    
    Features:
        - Multi-panel plots showing various metrics
        - Automatic plot updates during training
        - Metric history saving to JSON
        - Configurable plot aesthetics
    """
    
    def __init__(self, num_classes, save_dir='./training_plots', metrics_to_track=None):
        """
        Initialize the visualizer
        
        Args:
            num_classes (int): Number of segmentation classes
            save_dir (str): Directory to save plots and metrics
            metrics_to_track (list): List of metric names to track
        """
        self.num_classes = num_classes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Default metrics to track
        if metrics_to_track is None:
            metrics_to_track = [
                'train_loss', 'val_loss',
                'mean_dice_train', 'mean_dice_val',
                'class_dice_val', 'worst_k_dice',
                'learning_rate'
            ]
        self.metrics_to_track = metrics_to_track
        
        # Initialize metric storage
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'mean_dice_train': [],
            'mean_dice_val': [],
            'class_dice_val': [[] for _ in range(num_classes)],  # Per-class Dice
            'worst_k_dice': [],
            'learning_rate': [],
            'timestamp': str(datetime.now())
        }
        
        # Plot configuration
        self.fig = None
        self.axes = None
        
        print(f"✅ TrainingVisualizer initialized. Saving plots to: {self.save_dir}")
    
    def update(self, epoch, metrics):
        """
        Update metrics with new epoch data
        
        Args:
            epoch (int): Current epoch number
            metrics (dict): Dictionary containing metric values
                Expected keys: 'train_loss', 'val_loss', 'mean_dice_train',
                              'mean_dice_val', 'class_dice_val', 'worst_k_dice', 'lr'
        """
        self.history['epochs'].append(epoch)
        
        # Update scalar metrics
        for key in ['train_loss', 'val_loss', 'mean_dice_train', 
                    'mean_dice_val', 'worst_k_dice', 'learning_rate']:
            if key in metrics:
                self.history[key].append(metrics[key])
            elif key == 'learning_rate' and 'lr' in metrics:
                self.history['learning_rate'].append(metrics['lr'])
        
        # Update per-class Dice scores
        if 'class_dice_val' in metrics:
            class_dice = metrics['class_dice_val']
            for c in range(min(self.num_classes, len(class_dice))):
                self.history['class_dice_val'][c].append(class_dice[c])
    
    def plot_final_summary(self):
        """
        Generate and save final training summary plot with all metrics
        
        This method should be called once at the end of training or when early stopping occurs.
        It creates a comprehensive 6-panel visualization showing the training progress.
        """
        epochs = self.history['epochs']
        if len(epochs) == 0:
            print("⚠️  No data to plot yet")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Metrics (Epoch {epochs[-1]})', fontsize=16, fontweight='bold')
        
        # =====================================================================
        # Plot 1: Training and Validation Loss
        # =====================================================================
        ax = axes[0, 0]
        if len(self.history['train_loss']) > 0:
            ax.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if len(self.history['val_loss']) > 0:
            ax.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # =====================================================================
        # Plot 2: Mean Dice Score
        # =====================================================================
        ax = axes[0, 1]
        if len(self.history['mean_dice_train']) > 0:
            ax.plot(epochs, self.history['mean_dice_train'], 'b-', label='Train Dice', linewidth=2)
        if len(self.history['mean_dice_val']) > 0:
            ax.plot(epochs, self.history['mean_dice_val'], 'r-', label='Val Dice', linewidth=2)
        ax.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Target (0.7)')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Dice Score', fontsize=12)
        ax.set_title('Mean Dice Score', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # =====================================================================
        # Plot 3: Per-Class Dice Scores (Validation)
        # =====================================================================
        ax = axes[0, 2]
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
        for c in range(self.num_classes):
            if len(self.history['class_dice_val'][c]) > 0:
                ax.plot(epochs, self.history['class_dice_val'][c], 
                       color=colors[c], label=f'Class {c}', linewidth=2)
        ax.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Target (0.7)')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Dice Score', fontsize=12)
        ax.set_title('Per-Class Dice (Validation)', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # =====================================================================
        # Plot 4: Worst-K Class Dice Score
        # =====================================================================
        ax = axes[1, 0]
        if len(self.history['worst_k_dice']) > 0:
            ax.plot(epochs, self.history['worst_k_dice'], 'purple', linewidth=2, marker='o')
        ax.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Target (0.7)')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Min Dice Score', fontsize=12)
        ax.set_title('Worst-K Class Performance', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        # =====================================================================
        # Plot 5: Learning Rate
        # =====================================================================
        ax = axes[1, 1]
        if len(self.history['learning_rate']) > 0:
            ax.plot(epochs, self.history['learning_rate'], 'orange', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # =====================================================================
        # Plot 6: Training Summary Statistics
        # =====================================================================
        ax = axes[1, 2]
        ax.axis('off')
        
        # Get latest metrics
        latest_train_loss = self.history['train_loss'][-1] if self.history['train_loss'] else 0
        latest_val_loss = self.history['val_loss'][-1] if self.history['val_loss'] else 0
        latest_train_dice = self.history['mean_dice_train'][-1] if self.history['mean_dice_train'] else 0
        latest_val_dice = self.history['mean_dice_val'][-1] if self.history['mean_dice_val'] else 0
        latest_worst_k = self.history['worst_k_dice'][-1] if self.history['worst_k_dice'] else 0
        latest_lr = self.history['learning_rate'][-1] if self.history['learning_rate'] else 0
        
        # Best metrics
        best_val_dice = max(self.history['mean_dice_val']) if self.history['mean_dice_val'] else 0
        best_worst_k = max(self.history['worst_k_dice']) if self.history['worst_k_dice'] else 0
        
        summary_text = f"""
        TRAINING SUMMARY
        ─────────────────────────────
        Epoch: {epochs[-1]}
        
        Latest Metrics:
          • Train Loss: {latest_train_loss:.4f}
          • Val Loss: {latest_val_loss:.4f}
          • Train Dice: {latest_train_dice:.4f}
          • Val Dice: {latest_val_dice:.4f}
          • Worst-K Dice: {latest_worst_k:.4f}
          • Learning Rate: {latest_lr:.2e}
        
        Best Metrics:
          • Best Val Dice: {best_val_dice:.4f}
          • Best Worst-K: {best_worst_k:.4f}
        
        Status:
          • Target Dice ≥ 0.7: {'✅ ACHIEVED' if latest_val_dice >= 0.7 else '⏳ In Progress'}
        """
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # Save the final comprehensive plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = self.save_dir / f'training_summary_{timestamp}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        print(f"\n{'='*70}")
        print(f"✓ Training summary plot saved to: {plot_path}")
        print(f"{'='*70}\n")
        
        plt.close(fig)
    
    def save_history(self, filename='training_history.json'):
        """
        Save metric history to JSON file
        
        Args:
            filename (str): Output filename
        """
        filepath = self.save_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                if key == 'class_dice_val':
                    # Handle nested list
                    history_serializable[key] = [[float(v) if not np.isnan(v) else None 
                                                 for v in class_list] 
                                                for class_list in value]
                else:
                    history_serializable[key] = [float(v) if not np.isnan(v) else None 
                                                for v in value]
            else:
                history_serializable[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"💾 Training history saved: {filepath}")
    
    def load_history(self, filename='training_history.json'):
        """
        Load metric history from JSON file
        
        Args:
            filename (str): Input filename
        """
        filepath = self.save_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  History file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            self.history = json.load(f)
        
        print(f"✅ Training history loaded: {filepath}")
    
    def generate_final_report(self):
        """
        Generate a comprehensive final training report with all plots
        """
        if len(self.history['epochs']) == 0:
            print("⚠️  No training data to generate report")
            return
        
        # Save history
        self.save_history()
        
        # Generate text report
        report_path = self.save_dir / 'training_report.txt'
        
        epochs = self.history['epochs']
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Training Date: {self.history['timestamp']}\n")
            f.write(f"Total Epochs: {len(epochs)}\n")
            f.write(f"Number of Classes: {self.num_classes}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("FINAL METRICS\n")
            f.write("-"*80 + "\n")
            if self.history['train_loss']:
                f.write(f"Final Train Loss: {self.history['train_loss'][-1]:.4f}\n")
            if self.history['val_loss']:
                f.write(f"Final Val Loss: {self.history['val_loss'][-1]:.4f}\n")
            if self.history['mean_dice_train']:
                f.write(f"Final Train Dice: {self.history['mean_dice_train'][-1]:.4f}\n")
            if self.history['mean_dice_val']:
                f.write(f"Final Val Dice: {self.history['mean_dice_val'][-1]:.4f}\n")
            if self.history['worst_k_dice']:
                f.write(f"Final Worst-K Dice: {self.history['worst_k_dice'][-1]:.4f}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("BEST METRICS\n")
            f.write("-"*80 + "\n")
            if self.history['val_loss']:
                f.write(f"Best Val Loss: {min(self.history['val_loss']):.4f}\n")
            if self.history['mean_dice_val']:
                best_dice = max(self.history['mean_dice_val'])
                best_epoch = epochs[self.history['mean_dice_val'].index(best_dice)]
                f.write(f"Best Val Dice: {best_dice:.4f} (Epoch {best_epoch})\n")
            if self.history['worst_k_dice']:
                best_worst_k = max(self.history['worst_k_dice'])
                best_epoch = epochs[self.history['worst_k_dice'].index(best_worst_k)]
                f.write(f"Best Worst-K Dice: {best_worst_k:.4f} (Epoch {best_epoch})\n\n")
            
            f.write("-"*80 + "\n")
            f.write("PER-CLASS PERFORMANCE (FINAL VALIDATION)\n")
            f.write("-"*80 + "\n")
            for c in range(self.num_classes):
                if self.history['class_dice_val'][c]:
                    final_dice = self.history['class_dice_val'][c][-1]
                    f.write(f"Class {c}: {final_dice:.4f}\n")
        
        print(f"📄 Training report saved: {report_path}")
        print("="*80)
        print("✅ Final report generated successfully!")
        print("="*80)


if __name__ == "__main__":
    # Test the visualizer
    print("Testing TrainingVisualizer...")
    
    visualizer = TrainingVisualizer(num_classes=6, save_dir='./test_plots')
    
    # Simulate training for 20 epochs
    for epoch in range(1, 21):
        metrics = {
            'train_loss': 1.0 - (epoch * 0.03) + np.random.rand() * 0.1,
            'val_loss': 1.0 - (epoch * 0.025) + np.random.rand() * 0.1,
            'mean_dice_train': 0.3 + (epoch * 0.03) + np.random.rand() * 0.05,
            'mean_dice_val': 0.3 + (epoch * 0.025) + np.random.rand() * 0.05,
            'class_dice_val': [0.3 + (epoch * 0.02 * (c+1)/6) + np.random.rand() * 0.05 
                              for c in range(6)],
            'worst_k_dice': 0.2 + (epoch * 0.02) + np.random.rand() * 0.05,
            'lr': 1e-3 * (0.95 ** epoch)
        }
        
        visualizer.update(epoch, metrics)
        
        if epoch % 5 == 0:
            visualizer.plot(save=True)
    
    # Generate final report
    visualizer.generate_final_report()
    
    print("\n✅ Test complete! Check './test_plots' directory for outputs.")
