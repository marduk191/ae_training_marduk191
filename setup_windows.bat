@echo off
REM Setup script for Windows

echo ========================================
echo FLUX.1 VAE Training Setup for Windows
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python is installed
python --version
echo.

REM Check if CUDA is available
echo Checking for CUDA support...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>nul
if errorlevel 1 (
    echo [INFO] PyTorch not yet installed
) else (
    echo [OK] PyTorch already installed
)
echo.

REM Create virtual environment (optional but recommended)
echo Do you want to create a virtual environment? (y/n)
set /p create_venv=
if /i "%create_venv%"=="y" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment created and activated
    echo.
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install PyTorch
echo ========================================
echo PyTorch Installation
echo ========================================
echo.
echo Choose installation option:
echo 1. CUDA 11.8 (NVIDIA GPU - Recommended)
echo 2. CUDA 12.1 (NVIDIA GPU - Latest)
echo 3. CPU only (No GPU)
echo.
set /p torch_option="Enter option (1-3): "

if "%torch_option%"=="1" (
    echo Installing PyTorch with CUDA 11.8...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else if "%torch_option%"=="2" (
    echo Installing PyTorch with CUDA 12.1...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else if "%torch_option%"=="3" (
    echo Installing PyTorch (CPU only)...
    pip install torch torchvision
) else (
    echo Invalid option. Installing default PyTorch...
    pip install torch torchvision
)
echo.

REM Install other requirements
echo Installing other dependencies...
pip install numpy pillow tqdm tensorboard albumentations pytest black flake8
echo.

REM Create directory structure
echo Creating directory structure...
if not exist "data\train" mkdir data\train
if not exist "outputs" mkdir outputs
echo [OK] Directories created
echo.

REM Verify installation
echo ========================================
echo Verifying Installation
echo ========================================
echo.
python -c "import torch; import torchvision; import numpy; import PIL; print('All packages imported successfully!'); print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
if errorlevel 1 (
    echo [ERROR] Some packages failed to import
    pause
    exit /b 1
)
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Place your training images in the 'data/train' folder
echo 2. Run 'python prepare_data.py --mode validate --input_dir ./data/train' to validate your dataset
echo 3. Run 'train_example.bat' or use the command from README.md to start training
echo 4. Monitor training with: tensorboard --logdir ./outputs/experiment_1/logs
echo.
echo For more information, see README.md
echo.
pause
