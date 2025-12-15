#!/bin/bash
# Optimized training script for NVIDIA RTX 5090
# 32GB VRAM - Blackwell Architecture

echo "=========================================="
echo "RTX 5090 Optimized VAE Training"
echo "=========================================="
echo ""
echo "GPU: NVIDIA RTX 5090 (32GB VRAM)"
echo "CUDA: 12.8+ (Recommended)"
echo "Architecture: Blackwell"
echo ""

# Configuration options
echo "Choose training profile:"
echo "1. Ultra Quality (2048x2048, batch=4) - ~30GB VRAM"
echo "2. High Quality (1536x1536, batch=8) - ~28GB VRAM"
echo "3. Standard Quality (1024x1024, batch=16) - ~26GB VRAM [Recommended]"
echo "4. Fast Training (768x768, batch=24) - ~22GB VRAM"
echo "5. Custom settings"
echo ""
read -p "Enter option (1-5): " profile_option

case $profile_option in
    1)
        echo "Ultra Quality: 2048x2048 resolution"
        IMAGE_SIZE=2048
        BATCH_SIZE=4
        NUM_WORKERS=8
        ;;
    2)
        echo "High Quality: 1536x1536 resolution"
        IMAGE_SIZE=1536
        BATCH_SIZE=8
        NUM_WORKERS=8
        ;;
    3)
        echo "Standard Quality: 1024x1024 resolution [Recommended]"
        IMAGE_SIZE=1024
        BATCH_SIZE=16
        NUM_WORKERS=8
        ;;
    4)
        echo "Fast Training: 768x768 resolution"
        IMAGE_SIZE=768
        BATCH_SIZE=24
        NUM_WORKERS=8
        ;;
    5)
        echo "Custom settings mode"
        read -p "Image size (e.g., 1024): " IMAGE_SIZE
        read -p "Batch size (e.g., 16): " BATCH_SIZE
        read -p "Number of workers (e.g., 8): " NUM_WORKERS
        ;;
    *)
        echo "Invalid option. Using Standard Quality."
        IMAGE_SIZE=1024
        BATCH_SIZE=16
        NUM_WORKERS=8
        ;;
esac

echo ""
echo "Training Configuration:"
echo "  Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Workers: ${NUM_WORKERS}"
echo "  Mixed Precision: Enabled"
echo "  Gradient Checkpointing: Disabled (plenty of VRAM!)"
echo ""

read -p "Start training? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Start training with RTX 5090 optimizations
python train_vae.py \
  --data_path ./data/train \
  --output_dir ./outputs/rtx5090_${IMAGE_SIZE}x${IMAGE_SIZE} \
  --image_size ${IMAGE_SIZE} \
  --batch_size ${BATCH_SIZE} \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --z_channels 16 \
  --kl_weight 1e-6 \
  --perceptual_weight 1.0 \
  --mixed_precision \
  --gradient_clip 1.0 \
  --num_workers ${NUM_WORKERS} \
  --checkpoint_freq 1 \
  --save_samples_freq 100

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "View results with TensorBoard:"
echo "  tensorboard --logdir ./outputs/rtx5090_${IMAGE_SIZE}x${IMAGE_SIZE}/logs"
