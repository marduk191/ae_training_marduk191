#!/bin/bash
# Example training script for FLUX.1 VAE

# Basic training with default settings
python train_vae.py \
  --data_path ./data/train \
  --output_dir ./outputs/experiment_1 \
  --batch_size 4 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --image_size 512 \
  --mixed_precision \
  --num_workers 4

# Advanced training with custom settings
# python train_vae.py \
#   --data_path ./data/train \
#   --output_dir ./outputs/experiment_2 \
#   --batch_size 8 \
#   --num_epochs 200 \
#   --learning_rate 5e-5 \
#   --image_size 768 \
#   --z_channels 16 \
#   --kl_weight 1e-6 \
#   --perceptual_weight 1.0 \
#   --mixed_precision \
#   --gradient_clip 1.0 \
#   --num_workers 8

# Resume training from checkpoint
# python train_vae.py \
#   --data_path ./data/train \
#   --output_dir ./outputs/experiment_1 \
#   --resume_from ./outputs/experiment_1/checkpoints/checkpoint_epoch_50.pt \
#   --batch_size 4 \
#   --num_epochs 100
