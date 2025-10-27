# Installation Guide - 3D Medical Image Segmentation

## Quick Installation (Windows)

### Step 1: Install Python

Download and install Python 3.8 or higher from https://www.python.org/downloads/

During installation, **check "Add Python to PATH"**

### Step 2: Create Virtual Environment

Open PowerShell or Command Prompt:

```powershell
# Navigate to project directory
cd "C:\Users\17561\Desktop\new 3710\PatternAnalysis-2025-48316383\recognition\3D_ImprovedUNet_48316383"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate
```

### Step 3: Install PyTorch with CUDA (if you have NVIDIA GPU)

**Check your CUDA version first:**
```powershell
nvidia-smi
```

**For CUDA 11.8:**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only (no GPU):**
```powershell
pip install torch torchvision
```

### Step 4: Install Other Dependencies

```powershell
pip install -r requirements.txt
```

### Step 5: Verify Installation

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**Expected output:**
```
PyTorch: 2.x.x
CUDA Available: True
```

### Step 6: Test Import All Modules

```powershell
python -c "import nibabel, monai, scipy, matplotlib, tqdm; print('All dependencies installed successfully!')"
```

## Detailed Installation (Linux/Mac)

### Using venv

```bash
# Install Python 3.8+
sudo apt-get install python3.8 python3-pip  # Ubuntu/Debian
# brew install python@3.8  # macOS

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (check https://pytorch.org for your system)
pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt
```

### Using Conda (Recommended for Complex Environments)

```bash
# Create conda environment
conda create -n med_seg python=3.8
conda activate med_seg

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install nibabel monai scipy matplotlib tqdm
```

## Troubleshooting

### Issue 1: "pip is not recognized"

**Solution:**
```powershell
python -m pip install --upgrade pip
```

### Issue 2: "torch.cuda.is_available() returns False"

**Possible causes:**
- CUDA toolkit not installed
- Wrong PyTorch version (CPU-only)
- GPU drivers outdated

**Solutions:**
1. Update NVIDIA drivers: https://www.nvidia.com/Download/index.aspx
2. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
3. Reinstall PyTorch with correct CUDA version

### Issue 3: "ImportError: DLL load failed" (Windows)

**Solution:**
Install Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### Issue 4: Installation is very slow

**Solution:**
Use a mirror (China users):
```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Issue 5: Permission denied (Linux/Mac)

**Solution:**
Use `--user` flag:
```bash
pip install --user -r requirements.txt
```

## Verifying GPU Setup

Run this script to check GPU configuration:

```python
# test_gpu.py
import torch

print("="*60)
print("PyTorch GPU Configuration Check")
print("="*60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  No GPU detected. Training will be VERY slow on CPU.")
    print("Consider using Google Colab or Kaggle for free GPU access.")

# Test tensor operations
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(100, 100).to(device)
    y = torch.matmul(x, x)
    print("\n✅ Tensor operations working correctly!")
except Exception as e:
    print(f"\n❌ Error: {e}")

print("="*60)
```

Save as `test_gpu.py` and run:
```powershell
python test_gpu.py
```

## Docker Installation (Advanced)

For a containerized environment:

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train.py"]
```

Build and run:
```bash
docker build -t med-seg .
docker run --gpus all -v $(pwd):/workspace med-seg
```

## Google Colab Setup

If you don't have a GPU, use Google Colab (free GPU access):

1. Upload code to Google Drive
2. Open Colab notebook
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run installation:

```python
# In Colab cell
!pip install nibabel monai scipy matplotlib tqdm
!git clone <your-repo-url>
%cd 3D_ImprovedUNet_48316383

# Mount Google Drive for data
from google.colab import drive
drive.mount('/content/drive')

# Update data path in config.py
# Then run training
!python train.py
```

## Next Steps

After successful installation:

1. ✅ Run `python check_data.py` to verify dataset
2. ✅ Review and modify `config.py` for your setup
3. ✅ Start training with `python train.py`

## Getting Help

If installation issues persist:
1. Check PyTorch installation guide: https://pytorch.org/get-started/locally/
2. Check MONAI installation: https://docs.monai.io/en/stable/installation.html
3. Search error message on GitHub Issues or Stack Overflow
4. Ensure Python version is 3.8 or higher: `python --version`
