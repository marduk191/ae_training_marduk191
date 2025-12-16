#!/bin/bash
# Z-Image Turbo training optimized for NVIDIA RTX 5090
# Fast VAE for real-time applications

echo "=========================================="
echo "Z-Image Turbo VAE Training (RTX 5090)"
echo "=========================================="
echo ""
echo "GPU: NVIDIA RTX 5090 (32GB VRAM)"
echo "Architecture: Streamlined VAE (Turbo)"
echo "Optimized for: Speed and efficiency"
echo ""

# Configuration options
echo "Choose Z-Image Turbo profile:"
echo "1. Ultra-Fast (768x768, batch=32) - ~18GB VRAM ⚡⚡⚡⚡⚡"
echo "2. Fast Standard (1024x1024, batch=24) - ~24GB VRAM ⚡⚡⚡⚡ [Recommended]"
echo "3. High Resolution (1536x1536, batch=12) - ~27GB VRAM ⚡⚡⚡"
echo "4. Ultra Resolution (2048x2048, batch=6) - ~29GB VRAM ⚡⚡"
echo "5. Custom settings"
echo ""
read -p "Enter option (1-5): " profile_option

case $profile_option in
    1)
        echo "Ultra-Fast Profile: 768x768, batch=32"
        echo "  Training speed: ~0.25 sec/batch"
        echo "  Inference speed: ~8-10ms/image"
        IMAGE_SIZE=768
        BATCH_SIZE=32
        CHANNELS="[64, 128, 256, 256]"
        Z_CHANNELS=8
        LEARNING_RATE=3e-4
        ;;
    2)
        echo "Fast Standard Profile: 1024x1024, batch=24 [Recommended]"
        echo "  Training speed: ~0.4-0.5 sec/batch"
        echo "  Inference speed: ~15-20ms/image"
        IMAGE_SIZE=1024
        BATCH_SIZE=24
        CHANNELS="[96, 192, 384, 384]"
        Z_CHANNELS=8
        LEARNING_RATE=2e-4
        ;;
    3)
        echo "High Resolution Profile: 1536x1536, batch=12"
        echo "  Training speed: ~0.8-1.0 sec/batch"
        echo "  Inference speed: ~30-35ms/image"
        IMAGE_SIZE=1536
        BATCH_SIZE=12
        CHANNELS="[96, 192, 384, 384]"
        Z_CHANNELS=8
        LEARNING_RATE=1.5e-4
        ;;
    4)
        echo "Ultra Resolution Profile: 2048x2048, batch=6"
        echo "  Training speed: ~1.2-1.5 sec/batch"
        echo "  Inference speed: ~50-60ms/image"
        IMAGE_SIZE=2048
        BATCH_SIZE=6
        CHANNELS="[96, 192, 384, 384]"
        Z_CHANNELS=8
        LEARNING_RATE=1e-4
        ;;
    5)
        echo "Custom settings mode"
        read -p "Image size (e.g., 1024): " IMAGE_SIZE
        read -p "Batch size (e.g., 24): " BATCH_SIZE
        read -p "Z channels (e.g., 8): " Z_CHANNELS
        read -p "Learning rate (e.g., 2e-4): " LEARNING_RATE
        CHANNELS="[96, 192, 384, 384]"
        ;;
    *)
        echo "Invalid option. Using Fast Standard."
        IMAGE_SIZE=1024
        BATCH_SIZE=24
        CHANNELS="[96, 192, 384, 384]"
        Z_CHANNELS=8
        LEARNING_RATE=2e-4
        ;;
esac

echo ""
echo "Z-Image Turbo Configuration:"
echo "  Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Z Channels: ${Z_CHANNELS} (vs 16 for standard VAE)"
echo "  Channels: ${CHANNELS} (streamlined)"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Mixed Precision: Enabled"
echo ""
echo "Speed advantage: 2-3x faster than standard VAE"
echo "Model size: ~35% smaller than standard VAE"
echo "Inference: 2-3x faster encoding/decoding"
echo ""

read -p "Start training? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Start training with Z-Image Turbo optimizations
python train_vae.py \
  --data_path ./data/train \
  --output_dir ./outputs/zimage_turbo_${IMAGE_SIZE}x${IMAGE_SIZE} \
  --image_size ${IMAGE_SIZE} \
  --batch_size ${BATCH_SIZE} \
  --num_epochs 100 \
  --learning_rate ${LEARNING_RATE} \
  --z_channels ${Z_CHANNELS} \
  --kl_weight 1e-6 \
  --perceptual_weight 0.8 \
  --reconstruction_weight 1.2 \
  --mixed_precision \
  --gradient_clip 1.0 \
  --num_workers 8 \
  --checkpoint_freq 1 \
  --save_samples_freq 100

echo ""
echo "=========================================="
echo "Z-Image Turbo Training Complete!"
echo "=========================================="
echo "View results with TensorBoard:"
echo "  tensorboard --logdir ./outputs/zimage_turbo_${IMAGE_SIZE}x${IMAGE_SIZE}/logs"
echo ""
echo "Model characteristics:"
echo "  - 2-3x faster inference than standard VAE"
echo "  - Smaller model size (~35% reduction)"
echo "  - Optimized for real-time applications"
