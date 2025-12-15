@echo off
REM Example training script for FLUX.1 VAE on Windows

REM Basic training with default settings
python train_vae.py ^
  --data_path ./data/train ^
  --output_dir ./outputs/experiment_1 ^
  --batch_size 4 ^
  --num_epochs 100 ^
  --learning_rate 1e-4 ^
  --image_size 512 ^
  --mixed_precision ^
  --num_workers 4

REM Advanced training with custom settings (uncomment to use)
REM python train_vae.py ^
REM   --data_path ./data/train ^
REM   --output_dir ./outputs/experiment_2 ^
REM   --batch_size 8 ^
REM   --num_epochs 200 ^
REM   --learning_rate 5e-5 ^
REM   --image_size 768 ^
REM   --z_channels 16 ^
REM   --kl_weight 1e-6 ^
REM   --perceptual_weight 1.0 ^
REM   --mixed_precision ^
REM   --gradient_clip 1.0 ^
REM   --num_workers 8

REM Resume training from checkpoint (uncomment to use)
REM python train_vae.py ^
REM   --data_path ./data/train ^
REM   --output_dir ./outputs/experiment_1 ^
REM   --resume_from ./outputs/experiment_1/checkpoints/checkpoint_epoch_50.pt ^
REM   --batch_size 4 ^
REM   --num_epochs 100
