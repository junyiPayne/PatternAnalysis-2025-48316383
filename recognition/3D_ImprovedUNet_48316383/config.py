"""
Configuration File for 3D Medical Image Segmentation
=====================================================

This file contains all hyperparameters and settings for training,
validation, and inference of the 3D U-Net model.

"""

CONFIG = {
    # DATA CONFIGURATION
    'num_classes': 6,  # Number of segmentation classes
    'in_channels': 1,  # Input image channels (1 for grayscale MRI)
    'target_size': (128, 128, 64),  # Target volume size (H, W, D)
    'data_root': r"C:\Users\17561\Desktop\new 3710\data",  # Root data directory
    'img_subdir': "HipMRI_study_complete_release_v1/semantic_MRs_anon",  # Image subdirectory
    'label_subdir': "HipMRI_study_complete_release_v1/semantic_labels_anon",  # Label subdirectory
    'seed': 42,  # Random seed for reproducibility
    'split_ratio': (0.7, 0.15, 0.15),  # Train/Val/Test split ratio
    
    # MODEL CONFIGURATION
    'base_filters': 24,  # Base number of filters in U-Net (affects model size)
                         # Deeper model uses: 16 -> 32 -> 64 -> 128 -> 256 (bottleneck)
    
    # TRAINING CONFIGURATION
    'num_epochs': 20,  # Maximum number of training epochs
    'train_batch_size': 2,  # Training batch size
    'val_batch_size': 1,  # Validation batch size
    'learning_rate': 1e-3,  # Initial learning rate
    'weight_decay': 1e-5,  # L2 regularization
    'lr_decay_gamma': 0.985,  # Learning rate exponential decay factor
    'num_workers': 4,  # Number of DataLoader worker processes
    'use_preload': False,  # Whether to preload all data into memory
    'val_freq': 1,  # Validation frequency (validate every N epochs)
    'grad_accum_steps': 1,  # Gradient accumulation steps (simulate larger batch)
    
    # DYNAMIC WEIGHT ADJUSTER CONFIGURATION
    'weight_window_size': 3,  # Window size for tracking class performance
    'target_dice': 0.7,  # Target Dice score for class weight adjustment
    'min_improvement': 0.01,  # Minimum improvement threshold for weight adjustment
    
    # LOSS FUNCTION CONFIGURATION
    'loss_alpha': 1.0,  # Focal loss alpha parameter
    'loss_gamma': 1.0,  # Focal loss gamma parameter (focus on hard examples)
    'loss_dice_weight': 1.0,  # Weight for Dice loss component
    'loss_focal_weight': 20.0,  # Weight for Focal loss component
    'loss_smooth': 1.0,  # Smoothing factor for Dice loss
    
    # MIXED PRECISION TRAINING
    'use_amp': True,  # Enable Automatic Mixed Precision (saves memory, faster)
    
    # EARLY STOPPING CONFIGURATION
    'use_early_stopping': True,  # Enable early stopping
    'patience': 10,  # Number of epochs to wait for improvement (increased for deeper model)
    'min_delta': 0.001,  # Minimum improvement threshold (decreased for finer control)
    'num_worst_classes_to_track': 2,  # Track worst K non-background classes
    
    # DATA AUGMENTATION CONFIGURATION
    'augmentation': {
        'flip_prob': 0.5,  # Random flip probability
        'noise_std': 0.05,  # Gaussian noise standard deviation
        'intensity_scale_prob': 0.3,  # Random intensity scaling probability
    },
    
    # MONAI DATA LOADING
    'use_monai_loader': False,  # Use MONAI's CacheDataset for efficient loading (set to True to enable)
    'cache_rate': 0.5,  # Fraction of dataset to cache in memory (0.0 to 1.0)
    'num_cache_workers': 4,  # Workers for caching data
    
    # VISUALIZATION CONFIGURATION
    'enable_visualization': True,  # Enable training visualization
    'plot_dir': './training_plots',  # Directory to save plots
    'metrics_to_plot': [  # Metrics to track and plot
        'train_loss',
        'val_loss',
        'mean_dice_train',
        'mean_dice_val',
        'class_dice_val',  # Per-class Dice scores
        'worst_k_dice',  # Minimum Dice of worst K classes
        'learning_rate'
    ],
    
    # CHECKPOINT CONFIGURATION
    'checkpoint_dir': './checkpoints',  # Directory to save checkpoints
    'save_best_only': True,  # Save only the best model
    'save_frequency': 10,  # Save checkpoint every N epochs (if not best_only)
}