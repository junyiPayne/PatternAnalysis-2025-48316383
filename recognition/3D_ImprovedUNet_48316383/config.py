CONFIG = {
    'num_classes': 6,
    'target_size': (128, 128, 64),
    'batch_size': 2,
    'num_workers': 4,
    'base_filters': 16,
    'val_freq': 1,
    'grad_accum_steps': 1,
    'use_preload': False,
    'data_root': r"C:\Users\17561\Desktop\new 3710\data",
    'img_subdir': "HipMRI_study_complete_release_v1/semantic_MRs_anon",  # sub-directory
    'label_subdir': "HipMRI_study_complete_release_v1/semantic_labels_anon"  # label sub-directory
}