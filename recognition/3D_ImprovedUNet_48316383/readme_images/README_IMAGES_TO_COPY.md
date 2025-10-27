# README图片资源清单

## 📋 需要复制到此文件夹的图片

请将以下图片从指定位置复制到 `readme_images/` 文件夹：

### 1. Class Distribution & Weights (必需 - 说明动态权重调整机制)

```
来源: c:\Users\17561\Desktop\new 3710\
目标: readme_images/

复制文件:
- class_distribution.png       -> readme_images/class_distribution.png
- class_weights.png             -> readme_images/class_weights.png
```

**用途**: 展示数据集类别分布不平衡问题，以及动态权重调整机制如何根据分布计算权重


### 2. Training Summary Plots (必需 - 展示3个epoch配置的训练曲线)

```
来源: c:\Users\17561\Desktop\new 3710\training_plots\
目标: readme_images/

复制文件:
- training_summary_10epochs_20251028_043850.png  -> readme_images/training_summary_10epochs.png
- training_summary_20epochs_20251028_030918.png  -> readme_images/training_summary_20epochs.png
- training_summary_100epochs_20251028_041101.png -> readme_images/training_summary_100epochs.png
```

**用途**: 展示10/20/100轮训练的完整训练曲线（loss, Dice, per-class performance等）


### 3. Segmentation Comparisons (必需 - 同一病例在不同epoch下的分割对比)

#### 推荐选择: Case_015_Week0_LFOV (所有三个epoch都有)

```
来源: c:\Users\17561\Desktop\new 3710\visualizations_XXepochs\
目标: readme_images/comparisons/

Axial切面对比 (推荐 slice_128 - 中间切面):
- visualizations_10epochs/Case_015_Week0_LFOV_axial_slice_128.png  
  -> readme_images/comparisons/case015_axial_10epochs.png
  
- visualizations_20epochs/Case_015_Week0_LFOV_axial_slice_128.png  
  -> readme_images/comparisons/case015_axial_20epochs.png
  
- visualizations_42epochs/Case_015_Week0_LFOV_axial_slice_128.png  
  -> readme_images/comparisons/case015_axial_100epochs.png

Coronal切面对比 (推荐 slice_128 - 中间切面):
- visualizations_10epochs/Case_015_Week0_LFOV_coronal_slice_128.png  
  -> readme_images/comparisons/case015_coronal_10epochs.png
  
- visualizations_20epochs/Case_015_Week0_LFOV_coronal_slice_128.png  
  -> readme_images/comparisons/case015_coronal_20epochs.png
  
- visualizations_42epochs/Case_015_Week0_LFOV_coronal_slice_128.png  
  -> readme_images/comparisons/case015_coronal_100epochs.png
```

**用途**: 同一病例（Case_015）在不同训练轮数下的分割效果对比，展示训练进展


### 4. 额外推荐案例 (可选 - 展示不同病例的分割效果)

#### Case_031_Week3_LFOV - 中等难度案例

```
- visualizations_10epochs/Case_031_Week3_LFOV_axial_slice_128.png  
  -> readme_images/comparisons/case031_axial_10epochs.png
  
- visualizations_20epochs/Case_031_Week3_LFOV_axial_slice_128.png  
  -> readme_images/comparisons/case031_axial_20epochs.png
  
- visualizations_42epochs/Case_031_Week3_LFOV_axial_slice_128.png  
  -> readme_images/comparisons/case031_axial_100epochs.png
```

#### Case_040_Week5_LFOV - 另一个案例

```
- visualizations_10epochs/Case_040_Week5_LFOV_axial_slice_128.png  
  -> readme_images/comparisons/case040_axial_10epochs.png
  
- visualizations_20epochs/Case_040_Week5_LFOV_axial_slice_128.png  
  -> readme_images/comparisons/case040_axial_20epochs.png
  
- visualizations_42epochs/Case_040_Week5_LFOV_axial_slice_128.png  
  -> readme_images/comparisons/case040_axial_100epochs.png
```


## 📁 目标文件夹结构

```
readme_images/
├── class_distribution.png              # 类别分布图
├── class_weights.png                   # 动态权重图
├── training_summary_10epochs.png       # 10轮训练曲线
├── training_summary_20epochs.png       # 20轮训练曲线
├── training_summary_100epochs.png      # 100轮训练曲线
└── comparisons/                        # 分割对比图
    ├── case015_axial_10epochs.png
    ├── case015_axial_20epochs.png
    ├── case015_axial_100epochs.png
    ├── case015_coronal_10epochs.png
    ├── case015_coronal_20epochs.png
    └── case015_coronal_100epochs.png
```


## ✅ 复制命令 (PowerShell)

创建comparisons子文件夹：
```powershell
New-Item -ItemType Directory -Path "c:\Users\17561\Desktop\new 3710\PatternAnalysis-2025-48316383\recognition\3D_ImprovedUNet_48316383\readme_images\comparisons" -Force
```

### 复制Class Distribution & Weights:
```powershell
Copy-Item "c:\Users\17561\Desktop\new 3710\class_distribution.png" "c:\Users\17561\Desktop\new 3710\PatternAnalysis-2025-48316383\recognition\3D_ImprovedUNet_48316383\readme_images\class_distribution.png"

Copy-Item "c:\Users\17561\Desktop\new 3710\class_weights.png" "c:\Users\17561\Desktop\new 3710\PatternAnalysis-2025-48316383\recognition\3D_ImprovedUNet_48316383\readme_images\class_weights.png"
```

### 复制Training Summaries:
```powershell
Copy-Item "c:\Users\17561\Desktop\new 3710\training_plots\training_summary_10epochs_20251028_043850.png" "c:\Users\17561\Desktop\new 3710\PatternAnalysis-2025-48316383\recognition\3D_ImprovedUNet_48316383\readme_images\training_summary_10epochs.png"

Copy-Item "c:\Users\17561\Desktop\new 3710\training_plots\training_summary_20epochs_20251028_030918.png" "c:\Users\17561\Desktop\new 3710\PatternAnalysis-2025-48316383\recognition\3D_ImprovedUNet_48316383\readme_images\training_summary_20epochs.png"

Copy-Item "c:\Users\17561\Desktop\new 3710\training_plots\training_summary_100epochs_20251028_041101.png" "c:\Users\17561\Desktop\new 3710\PatternAnalysis-2025-48316383\recognition\3D_ImprovedUNet_48316383\readme_images\training_summary_100epochs.png"
```

### 复制Case 015对比图 (Axial):
```powershell
Copy-Item "c:\Users\17561\Desktop\new 3710\visualizations_10epochs\Case_015_Week0_LFOV_axial_slice_128.png" "c:\Users\17561\Desktop\new 3710\PatternAnalysis-2025-48316383\recognition\3D_ImprovedUNet_48316383\readme_images\comparisons\case015_axial_10epochs.png"

Copy-Item "c:\Users\17561\Desktop\new 3710\visualizations_20epochs\Case_015_Week0_LFOV_axial_slice_128.png" "c:\Users\17561\Desktop\new 3710\PatternAnalysis-2025-48316383\recognition\3D_ImprovedUNet_48316383\readme_images\comparisons\case015_axial_20epochs.png"

Copy-Item "c:\Users\17561\Desktop\new 3710\visualizations_42epochs\Case_015_Week0_LFOV_axial_slice_128.png" "c:\Users\17561\Desktop\new 3710\PatternAnalysis-2025-48316383\recognition\3D_ImprovedUNet_48316383\readme_images\comparisons\case015_axial_100epochs.png"
```

### 复制Case 015对比图 (Coronal):
```powershell
Copy-Item "c:\Users\17561\Desktop\new 3710\visualizations_10epochs\Case_015_Week0_LFOV_coronal_slice_128.png" "c:\Users\17561\Desktop\new 3710\PatternAnalysis-2025-48316383\recognition\3D_ImprovedUNet_48316383\readme_images\comparisons\case015_coronal_10epochs.png"

Copy-Item "c:\Users\17561\Desktop\new 3710\visualizations_20epochs\Case_015_Week0_LFOV_coronal_slice_128.png" "c:\Users\17561\Desktop\new 3710\PatternAnalysis-2025-48316383\recognition\3D_ImprovedUNet_48316383\readme_images\comparisons\case015_coronal_20epochs.png"

Copy-Item "c:\Users\17561\Desktop\new 3710\visualizations_42epochs\Case_015_Week0_LFOV_coronal_slice_128.png" "c:\Users\17561\Desktop\new 3710\PatternAnalysis-2025-48316383\recognition\3D_ImprovedUNet_48316383\readme_images\comparisons\case015_coronal_100epochs.png"
```


## 🎯 README中的引用路径

在README.md中，图片将使用以下相对路径引用：

```markdown
![Class Distribution](./readme_images/class_distribution.png)
![Class Weights](./readme_images/class_weights.png)
![Training Summary - 10 Epochs](./readme_images/training_summary_10epochs.png)
![Case 015 Axial - 10 Epochs](./readme_images/comparisons/case015_axial_10epochs.png)
```


## 📝 说明

1. **Case_015_Week0_LFOV** 在所有三个epoch配置中都有相同的切面，适合做对比
2. **Axial切面** (slice_128) 是水平方向的中间切面，通常能清楚显示所有器官
3. **Coronal切面** (slice_128) 是冠状方向的中间切面，提供不同视角
4. 所有图片都已重命名为易读格式，便于在README中引用
5. 图片将展示：原始图像、Ground Truth、预测结果的4面板对比

