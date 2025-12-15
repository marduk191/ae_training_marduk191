@echo off
REM Optimized training script for NVIDIA RTX 5090
REM 32GB VRAM - Blackwell Architecture

echo ==========================================
echo RTX 5090 Optimized VAE Training
echo ==========================================
echo.
echo GPU: NVIDIA RTX 5090 (32GB VRAM)
echo CUDA: 12.8+ (Recommended)
echo Architecture: Blackwell
echo.

REM Configuration options
echo Choose training profile:
echo 1. Ultra Quality (2048x2048, batch=4) - ~30GB VRAM
echo 2. High Quality (1536x1536, batch=8) - ~28GB VRAM
echo 3. Standard Quality (1024x1024, batch=16) - ~26GB VRAM [Recommended]
echo 4. Fast Training (768x768, batch=24) - ~22GB VRAM
echo 5. Custom settings
echo.
set /p profile_option="Enter option (1-5): "

if "%profile_option%"=="1" (
    echo Ultra Quality: 2048x2048 resolution
    set IMAGE_SIZE=2048
    set BATCH_SIZE=4
    set NUM_WORKERS=8
) else if "%profile_option%"=="2" (
    echo High Quality: 1536x1536 resolution
    set IMAGE_SIZE=1536
    set BATCH_SIZE=8
    set NUM_WORKERS=8
) else if "%profile_option%"=="3" (
    echo Standard Quality: 1024x1024 resolution [Recommended]
    set IMAGE_SIZE=1024
    set BATCH_SIZE=16
    set NUM_WORKERS=8
) else if "%profile_option%"=="4" (
    echo Fast Training: 768x768 resolution
    set IMAGE_SIZE=768
    set BATCH_SIZE=24
    set NUM_WORKERS=8
) else if "%profile_option%"=="5" (
    echo Custom settings mode
    set /p IMAGE_SIZE="Image size (e.g., 1024): "
    set /p BATCH_SIZE="Batch size (e.g., 16): "
    set /p NUM_WORKERS="Number of workers (e.g., 8): "
) else (
    echo Invalid option. Using Standard Quality.
    set IMAGE_SIZE=1024
    set BATCH_SIZE=16
    set NUM_WORKERS=8
)

echo.
echo Training Configuration:
echo   Image Size: %IMAGE_SIZE%x%IMAGE_SIZE%
echo   Batch Size: %BATCH_SIZE%
echo   Workers: %NUM_WORKERS%
echo   Mixed Precision: Enabled
echo   Gradient Checkpointing: Disabled (plenty of VRAM!)
echo.

set /p start_training="Start training? (y/n): "
if /i not "%start_training%"=="y" (
    exit /b 0
)

REM Start training with RTX 5090 optimizations
python train_vae.py ^
  --data_path ./data/train ^
  --output_dir ./outputs/rtx5090_%IMAGE_SIZE%x%IMAGE_SIZE% ^
  --image_size %IMAGE_SIZE% ^
  --batch_size %BATCH_SIZE% ^
  --num_epochs 100 ^
  --learning_rate 1e-4 ^
  --z_channels 16 ^
  --kl_weight 1e-6 ^
  --perceptual_weight 1.0 ^
  --mixed_precision ^
  --gradient_clip 1.0 ^
  --num_workers %NUM_WORKERS% ^
  --checkpoint_freq 1 ^
  --save_samples_freq 100

echo.
echo ==========================================
echo Training Complete!
echo ==========================================
echo View results with TensorBoard:
echo   tensorboard --logdir ./outputs/rtx5090_%IMAGE_SIZE%x%IMAGE_SIZE%/logs
