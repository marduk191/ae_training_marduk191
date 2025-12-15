# Windows Setup and Troubleshooting Guide

## Quick Setup

### Prerequisites

1. **Python 3.12+** - Download from [python.org](https://www.python.org/downloads/)
   - ✅ During installation, check "Add Python to PATH"
   - ✅ Enable "pip" installation
   - ⚠️ Python 3.12 or higher is **required** for full compatibility
   - 💡 Recommended: Python 3.12.x (latest stable)

2. **NVIDIA GPU (Optional but Recommended)**
   - Install latest GPU drivers from [nvidia.com](https://www.nvidia.com/Download/index.aspx)
   - CUDA will be installed automatically with PyTorch

3. **Git** (Optional) - Download from [git-scm.com](https://git-scm.com/download/win)

### Automated Setup

1. Open Command Prompt or PowerShell
2. Navigate to the project directory:
   ```cmd
   cd path\to\ae_training_marduk191
   ```
3. Run the setup script:
   ```cmd
   setup_windows.bat
   ```
4. Follow the prompts to install dependencies

## Manual Installation Steps

### Step 1: Verify Python

```cmd
python --version
```

Should show Python 3.12 or higher.

**Run the compatibility checker:**
```cmd
python check_compatibility.py
```

This will verify:
- Python version is 3.12+
- All required packages are installed
- CUDA/GPU is properly configured

### Step 2: Create Virtual Environment (Recommended)

```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your command prompt.

### Step 3: Install PyTorch

**For NVIDIA GPU:**
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```cmd
pip install torch torchvision
```

### Step 4: Install Other Dependencies

```cmd
pip install numpy pillow tqdm tensorboard albumentations
```

### Step 5: Verify Installation

```cmd
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Common Issues and Solutions

### Issue 1: "Python is not recognized"

**Problem**: Python is not in your system PATH.

**Solution**:
1. Reinstall Python and check "Add Python to PATH"
2. Or manually add Python to PATH:
   - Search for "Environment Variables" in Windows
   - Edit PATH variable
   - Add: `C:\Python3X\` and `C:\Python3X\Scripts\`

### Issue 2: CUDA Not Available

**Problem**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. **Check GPU**: Open Task Manager → Performance → GPU to verify NVIDIA GPU exists
2. **Update Drivers**: Download latest from [nvidia.com](https://www.nvidia.com/Download/index.aspx)
3. **Reinstall PyTorch with CUDA**:
   ```cmd
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
4. **Verify CUDA Version**: Some older GPUs may need older CUDA versions

### Issue 3: Out of Memory (OOM) Errors

**Problem**: Training crashes with CUDA out of memory error.

**Solutions**:
1. **Reduce batch size**:
   ```cmd
   python train_vae.py --batch_size 2 --data_path ./data/train
   ```
2. **Reduce image size**:
   ```cmd
   python train_vae.py --image_size 256 --data_path ./data/train
   ```
3. **Close other GPU applications** (Chrome, games, etc.)
4. **Enable gradient accumulation**:
   ```cmd
   python train_vae.py --batch_size 2 --accumulation_steps 2 --data_path ./data/train
   ```

### Issue 4: "Permission Denied" Errors

**Problem**: Cannot write to directories.

**Solutions**:
1. **Run as Administrator**: Right-click Command Prompt → Run as Administrator
2. **Check folder permissions**: Ensure you have write access to the project folder
3. **Use a different output directory**:
   ```cmd
   python train_vae.py --output_dir C:\Users\YourName\Documents\vae_output --data_path ./data/train
   ```

### Issue 5: Slow Data Loading

**Problem**: Training is very slow, stuck at data loading.

**Solutions**:
1. **Reduce workers**:
   ```cmd
   python train_vae.py --num_workers 0 --data_path ./data/train
   ```
   (Windows has issues with multiprocessing, start with 0 or 2)

2. **Move data to SSD**: If images are on HDD, copy to SSD
3. **Reduce dataset size**: Start with fewer images for testing

### Issue 6: Import Errors

**Problem**: `ModuleNotFoundError` when running scripts.

**Solutions**:
1. **Activate virtual environment**:
   ```cmd
   venv\Scripts\activate
   ```
2. **Reinstall dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```
3. **Check Python version**: Some packages require Python 3.8+

### Issue 7: Path Issues

**Problem**: File not found errors with paths.

**Solutions**:
1. **Use forward slashes** even on Windows:
   ```cmd
   python train_vae.py --data_path ./data/train
   ```
2. **Or use backslashes**:
   ```cmd
   python train_vae.py --data_path .\data\train
   ```
3. **Use absolute paths** if relative paths fail:
   ```cmd
   python train_vae.py --data_path C:\Users\YourName\project\data\train
   ```

## Performance Tips for Windows

### 1. Use Windows Terminal

Windows Terminal is faster and more reliable than cmd.exe:
- Download from Microsoft Store
- Better color support for progress bars
- Better Unicode support

### 2. Disable Windows Defender During Training

Windows Defender can slow down file I/O:
1. Go to Windows Security → Virus & threat protection
2. Add the project folder to exclusions
3. ⚠️ Only do this for trusted code!

### 3. Use Virtual Environment

Always use a virtual environment to avoid package conflicts:
```cmd
python -m venv venv
venv\Scripts\activate
```

### 4. Monitor GPU Usage

Use these tools to monitor GPU during training:
- **Task Manager**: Performance → GPU
- **NVIDIA System Monitor**: Installed with drivers
- **GPUtil**: Install with `pip install gputil`

### 5. Optimize Worker Count

Windows has different multiprocessing behavior than Linux:
- Start with `--num_workers 0` or `--num_workers 2`
- Test different values to find optimal performance
- Too many workers can actually slow things down on Windows

## Testing Your Setup

### Quick Test Script

Create a file `test_setup.py`:
```python
import torch
import torchvision
import numpy as np
from PIL import Image

print("=" * 50)
print("System Configuration")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Test CUDA
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.mm(x, y)
    print("✅ CUDA test passed")
else:
    print("⚠️ CUDA not available, will use CPU")

print("\n" + "=" * 50)
print("Setup is ready!")
print("=" * 50)
```

Run it:
```cmd
python test_setup.py
```

## Recommended Development Setup

1. **IDE**: VS Code with Python extension
2. **Terminal**: Windows Terminal
3. **Python**: 3.10 or 3.11 (good stability)
4. **Virtual Environment**: Always use one
5. **GPU Drivers**: Keep updated

## Getting Help

If you encounter issues not covered here:

1. Check the error message carefully
2. Search for the error on:
   - [PyTorch Forums](https://discuss.pytorch.org/)
   - [Stack Overflow](https://stackoverflow.com/questions/tagged/pytorch)
3. Include system information:
   ```cmd
   python --version
   python -c "import torch; print(torch.__version__)"
   nvidia-smi
   ```

## Additional Resources

- [PyTorch Windows Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [Windows Python Development Guide](https://docs.python.org/3/using/windows.html)
