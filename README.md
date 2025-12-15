# FLUX.1 VAE Training Script

A comprehensive Python implementation for training Variational Autoencoders (VAE) compatible with FLUX.1 and similar image generation models.

## Features

- **Production-Ready Architecture**: Robust VAE implementation with encoder, decoder, and latent space manipulation
- **Advanced Training**: Mixed precision training, gradient clipping, and learning rate scheduling
- **Flexible Configuration**: JSON-based configuration system with command-line overrides
- **Multiple Loss Functions**: Reconstruction loss, KL divergence, and perceptual loss (VGG-based)
- **Comprehensive Monitoring**: TensorBoard integration, sample visualization, and detailed metrics
- **Data Pipeline**: Efficient data loading with preprocessing, validation, and augmentation
- **Inference Tools**: Encoding, decoding, interpolation, and variation generation
- **Checkpointing**: Automatic checkpoint saving and resume capability

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ VRAM for 512x512 images with batch size 4

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ae_training_marduk191
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Organize your training images in a directory:
```
data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
```

#### Validate Dataset
```bash
python prepare_data.py \
  --mode validate \
  --input_dir ./data/train \
  --output_file validation_report.json
```

#### Preprocess Images
```bash
python prepare_data.py \
  --mode preprocess \
  --input_dir ./data/raw \
  --output_dir ./data/train \
  --target_size 512 \
  --resize_mode cover
```

#### Analyze Dataset
```bash
python prepare_data.py \
  --mode analyze \
  --input_dir ./data/train \
  --output_file dataset_stats.json
```

### 2. Train the VAE

Basic training:
```bash
python train_vae.py \
  --data_path ./data/train \
  --output_dir ./outputs/my_vae \
  --batch_size 4 \
  --num_epochs 100 \
  --mixed_precision
```

Advanced training with custom parameters:
```bash
python train_vae.py \
  --data_path ./data/train \
  --output_dir ./outputs/my_vae \
  --batch_size 8 \
  --num_epochs 200 \
  --learning_rate 5e-5 \
  --image_size 768 \
  --z_channels 16 \
  --kl_weight 1e-6 \
  --perceptual_weight 1.0 \
  --gradient_clip 1.0 \
  --num_workers 8 \
  --mixed_precision
```

### 3. Monitor Training

View training progress with TensorBoard:
```bash
tensorboard --logdir ./outputs/my_vae/logs
```

### 4. Resume Training

```bash
python train_vae.py \
  --data_path ./data/train \
  --output_dir ./outputs/my_vae \
  --resume_from ./outputs/my_vae/checkpoints/checkpoint_epoch_50.pt
```

## Inference

### Reconstruct Images

```bash
python inference.py \
  --checkpoint ./outputs/my_vae/checkpoints/best_model.pt \
  --mode reconstruct \
  --input ./test_image.jpg \
  --output ./reconstructed.jpg
```

### Encode Images to Latents

```bash
python inference.py \
  --checkpoint ./outputs/my_vae/checkpoints/best_model.pt \
  --mode encode \
  --input ./image.jpg \
  --output ./latent.pt
```

### Interpolate Between Images

```bash
python inference.py \
  --checkpoint ./outputs/my_vae/checkpoints/best_model.pt \
  --mode interpolate \
  --input ./image1.jpg \
  --input2 ./image2.jpg \
  --output ./interpolations/ \
  --steps 10
```

### Generate Variations

```bash
python inference.py \
  --checkpoint ./outputs/my_vae/checkpoints/best_model.pt \
  --mode variations \
  --input ./image.jpg \
  --output ./variations/ \
  --num_variations 5 \
  --temperature 0.5
```

## Configuration

### Model Architecture

- `latent_dim`: Latent dimension (default: 16)
- `z_channels`: Number of latent channels (default: 16)
- `channels`: Encoder/decoder channel progression (default: [128, 256, 512, 512])
- `image_size`: Input image size (default: 512)
- `in_channels`: Number of input channels (default: 3 for RGB)

### Training Hyperparameters

- `batch_size`: Batch size (default: 4)
- `num_epochs`: Number of training epochs (default: 100)
- `learning_rate`: Initial learning rate (default: 1e-4)
- `weight_decay`: Weight decay for regularization (default: 1e-5)
- `warmup_epochs`: Number of warmup epochs (default: 5)

### Loss Weights

- `kl_weight`: KL divergence weight (default: 1e-6)
- `perceptual_weight`: Perceptual loss weight (default: 1.0)
- `reconstruction_weight`: Reconstruction loss weight (default: 1.0)

### Training Settings

- `mixed_precision`: Enable mixed precision training (default: true)
- `gradient_clip`: Gradient clipping threshold (default: 1.0)
- `accumulation_steps`: Gradient accumulation steps (default: 1)
- `num_workers`: Data loading workers (default: 4)

## Project Structure

```
ae_training_marduk191/
├── train_vae.py           # Main training script
├── inference.py           # Inference utilities
├── prepare_data.py        # Data preparation tools
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── configs/
│   └── example_config.json  # Example configuration
├── train_example.sh      # Example training script
├── data/
│   └── train/           # Training images
└── outputs/
    └── my_vae/
        ├── checkpoints/ # Model checkpoints
        ├── samples/     # Sample reconstructions
        └── logs/        # TensorBoard logs
```

## Architecture Details

### Encoder
- Multiple residual blocks with group normalization
- Self-attention blocks at lower resolutions
- Progressive downsampling
- Output: Mean and log-variance of latent distribution

### Decoder
- Symmetric architecture to encoder
- Progressive upsampling
- Residual blocks with self-attention
- Output: Reconstructed image

### Loss Function
The training loss combines three components:

1. **Reconstruction Loss** (MSE): Measures pixel-wise difference
2. **KL Divergence**: Regularizes latent space distribution
3. **Perceptual Loss** (VGG): Captures high-level semantic similarity

Total loss = `reconstruction_weight * L_recon + kl_weight * L_kl + perceptual_weight * L_perceptual`

## Tips for Training

### Memory Optimization
- Reduce `batch_size` if running out of memory
- Use `mixed_precision` for 2x memory reduction
- Reduce `image_size` (e.g., 256 or 384)
- Use `accumulation_steps` to simulate larger batches

### Quality Improvements
- Increase `num_epochs` for better convergence
- Adjust loss weights based on validation results
- Use higher `image_size` for finer details
- Collect diverse training data (10K+ images recommended)

### Training Speed
- Increase `num_workers` for faster data loading
- Use SSD for data storage
- Enable `mixed_precision` for 1.5-2x speedup
- Use multiple GPUs (modify script for DDP)

## Troubleshooting

### NaN Loss
- Reduce learning rate
- Increase gradient clipping
- Check for corrupted images in dataset
- Reduce KL weight

### Poor Reconstruction Quality
- Increase perceptual loss weight
- Train for more epochs
- Increase model capacity (more channels)
- Check data preprocessing

### Mode Collapse
- Increase KL weight
- Add dropout or other regularization
- Use learning rate warmup
- Ensure diverse training data

## Advanced Usage

### Custom Configuration File

Create a JSON config file:
```json
{
  "latent_dim": 16,
  "channels": [128, 256, 512, 512],
  "image_size": 512,
  "batch_size": 4,
  "num_epochs": 100,
  "learning_rate": 1e-4,
  "kl_weight": 1e-6,
  "mixed_precision": true
}
```

### Multi-GPU Training

Modify the training script to use PyTorch DistributedDataParallel:
```python
# Add to train_vae.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')
model = DDP(model, device_ids=[local_rank])
```

### Integration with FLUX.1

The trained VAE can be integrated with FLUX.1 or similar models:
```python
from train_vae import VAE, TrainingConfig

# Load trained VAE
checkpoint = torch.load('best_model.pt')
config = TrainingConfig(**checkpoint['config'])
vae = VAE(config)
vae.load_state_dict(checkpoint['model_state_dict'])

# Use for encoding images
latents = vae.encode(images)

# Or decoding latents
reconstructed = vae.decode(latents)
```

## License

This project is provided as-is for research and educational purposes.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{flux_vae_training,
  title={FLUX.1 VAE Training Script},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ae_training_marduk191}
}
```

## Acknowledgments

- Inspired by the FLUX.1 architecture
- VAE architecture based on modern diffusion model practices
- Perceptual loss using VGG16 features

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Note**: This is a research implementation. For production use, consider additional optimization, validation, and testing.
