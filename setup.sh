#!/bin/bash
# Setup script for Linux/macOS - Python 3.12+ VAE Training

set -e

echo "========================================"
echo "FLUX.1 VAE Training Setup"
echo "Python 3.12+ Required"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 is not installed"
    echo "Please install Python 3.12+ from:"
    echo "  - Ubuntu/Debian: sudo apt install python3.12 python3.12-venv python3-pip"
    echo "  - macOS: brew install python@3.12"
    echo "  - Or download from: https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Python is installed"
python3 --version
echo ""

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_VERSION="3.12"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "⚠️  WARNING: Python ${REQUIRED_VERSION}+ is required"
    echo "Current version: ${PYTHON_VERSION}"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Python version is ${PYTHON_VERSION} (compatible)"
fi
echo ""

# Ask about virtual environment
read -p "Create a virtual environment? (recommended) (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated"
    echo ""
fi

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip
echo ""

# PyTorch installation
echo "========================================"
echo "PyTorch Installation"
echo "========================================"
echo ""
echo "Choose installation option:"
echo "1. CUDA 11.8 (NVIDIA GPU - Most Compatible)"
echo "2. CUDA 12.1 (NVIDIA GPU - Latest)"
echo "3. CUDA 12.4 (NVIDIA GPU - Newest, Python 3.12+ optimized)"
echo "4. ROCm 5.7 (AMD GPU)"
echo "5. CPU only (No GPU)"
echo ""
read -p "Enter option (1-5): " torch_option

case $torch_option in
    1)
        echo "Installing PyTorch with CUDA 11.8..."
        pip install torch>=2.1.0 torchvision>=0.16.0 --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        echo "Installing PyTorch with CUDA 12.1..."
        pip install torch>=2.1.0 torchvision>=0.16.0 --index-url https://download.pytorch.org/whl/cu121
        ;;
    3)
        echo "Installing PyTorch with CUDA 12.4 (Recommended for Python 3.12+)..."
        pip install torch>=2.1.0 torchvision>=0.16.0 --index-url https://download.pytorch.org/whl/cu124
        ;;
    4)
        echo "Installing PyTorch with ROCm 5.7..."
        pip install torch>=2.1.0 torchvision>=0.16.0 --index-url https://download.pytorch.org/whl/rocm5.7
        ;;
    5)
        echo "Installing PyTorch (CPU only)..."
        pip install torch>=2.1.0 torchvision>=0.16.0
        ;;
    *)
        echo "Invalid option. Installing default PyTorch..."
        pip install torch>=2.1.0 torchvision>=0.16.0
        ;;
esac
echo ""

# Install other dependencies
echo "Installing other dependencies..."
pip install numpy>=1.26.0 pillow>=10.0.0 tqdm>=4.66.0 tensorboard>=2.15.0 albumentations>=1.4.0
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/train
mkdir -p outputs
echo "✅ Directories created"
echo ""

# Run compatibility check
echo "========================================"
echo "Verifying Installation"
echo "========================================"
echo ""
python3 check_compatibility.py

# Final instructions
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. If you created a virtual environment, activate it:"
echo "   source venv/bin/activate"
echo ""
echo "2. Place your training images in the 'data/train' folder"
echo ""
echo "3. Validate your dataset:"
echo "   python prepare_data.py --mode validate --input_dir ./data/train"
echo ""
echo "4. Start training:"
echo "   bash train_example.sh"
echo "   OR"
echo "   python train_vae.py --data_path ./data/train --output_dir ./outputs/my_vae --batch_size 4 --num_epochs 100 --mixed_precision"
echo ""
echo "5. Monitor training with TensorBoard:"
echo "   tensorboard --logdir ./outputs/my_vae/logs"
echo ""
echo "For more information, see README.md"
echo ""
