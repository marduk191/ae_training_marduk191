#!/usr/bin/env python3
"""
Comprehensive FLUX.1 VAE Training Script
Supports training variational autoencoders for image generation models.

Requires Python 3.12+
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for VAE training"""

    def __init__(self, **kwargs):
        # Model architecture
        self.latent_dim = kwargs.get('latent_dim', 16)
        self.channels = kwargs.get('channels', [128, 256, 512, 512])
        self.image_size = kwargs.get('image_size', 512)
        self.in_channels = kwargs.get('in_channels', 3)
        self.z_channels = kwargs.get('z_channels', 16)

        # Training hyperparameters
        self.batch_size = kwargs.get('batch_size', 4)
        self.num_epochs = kwargs.get('num_epochs', 100)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.warmup_epochs = kwargs.get('warmup_epochs', 5)

        # Loss weights
        self.kl_weight = kwargs.get('kl_weight', 1e-6)
        self.perceptual_weight = kwargs.get('perceptual_weight', 1.0)
        self.reconstruction_weight = kwargs.get('reconstruction_weight', 1.0)

        # Data
        self.data_path = kwargs.get('data_path', './data')
        self.num_workers = kwargs.get('num_workers', 4)

        # Training settings
        self.mixed_precision = kwargs.get('mixed_precision', True)
        self.gradient_clip = kwargs.get('gradient_clip', 1.0)
        self.accumulation_steps = kwargs.get('accumulation_steps', 1)

        # Checkpointing
        self.output_dir = kwargs.get('output_dir', './outputs')
        self.checkpoint_freq = kwargs.get('checkpoint_freq', 1)
        self.save_samples_freq = kwargs.get('save_samples_freq', 100)

        # Resume training
        self.resume_from = kwargs.get('resume_from', None)

        # Validation
        self.val_split = kwargs.get('val_split', 0.1)
        self.val_freq = kwargs.get('val_freq', 1)

        # Device
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    def save(self, path: str):
        """Save configuration to JSON"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class ImageDataset(Dataset):
    """Dataset for loading images for VAE training"""

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (512, 512), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, img_path


class ResidualBlock(nn.Module):
    """Residual block with group normalization"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.activation = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x + residual


class AttentionBlock(nn.Module):
    """Self-attention block for VAE"""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        residual = x
        x = self.norm(x)

        b, c, h, w = x.shape
        q = self.q(x).reshape(b, c, h * w).permute(0, 2, 1)
        k = self.k(x).reshape(b, c, h * w)
        v = self.v(x).reshape(b, c, h * w).permute(0, 2, 1)

        attn = torch.bmm(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        out = self.proj_out(out)

        return out + residual


class Encoder(nn.Module):
    """VAE Encoder"""

    def __init__(self, in_channels: int, channels: list, z_channels: int):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels)):
            in_ch = channels[i]
            out_ch = channels[min(i + 1, len(channels) - 1)]

            block = nn.Sequential(
                ResidualBlock(in_ch, out_ch),
                ResidualBlock(out_ch, out_ch)
            )
            self.down_blocks.append(block)

            # Add attention at lower resolutions
            if i >= len(channels) - 2:
                self.down_blocks.append(AttentionBlock(out_ch))

        # Middle blocks
        self.mid_block1 = ResidualBlock(channels[-1], channels[-1])
        self.mid_attn = AttentionBlock(channels[-1])
        self.mid_block2 = ResidualBlock(channels[-1], channels[-1])

        # Output projection
        self.norm_out = nn.GroupNorm(32, channels[-1])
        self.conv_out = nn.Conv2d(channels[-1], 2 * z_channels, 3, padding=1)

        self.downsample = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.conv_in(x)

        # Downsampling
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            # Downsample after every 2 residual blocks (except attention blocks)
            if i % 3 == 1:  # After residual blocks, before attention
                x = self.downsample(x)

        # Middle
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        # Output
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x


class Decoder(nn.Module):
    """VAE Decoder"""

    def __init__(self, z_channels: int, channels: list, out_channels: int):
        super().__init__()
        self.conv_in = nn.Conv2d(z_channels, channels[-1], 3, padding=1)

        # Middle blocks
        self.mid_block1 = ResidualBlock(channels[-1], channels[-1])
        self.mid_attn = AttentionBlock(channels[-1])
        self.mid_block2 = ResidualBlock(channels[-1], channels[-1])

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))

        for i in range(len(reversed_channels)):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[min(i + 1, len(reversed_channels) - 1)]

            # Add attention at lower resolutions
            if i < 2:
                self.up_blocks.append(AttentionBlock(in_ch))

            block = nn.Sequential(
                ResidualBlock(in_ch, out_ch),
                ResidualBlock(out_ch, out_ch)
            )
            self.up_blocks.append(block)

        # Output projection
        self.norm_out = nn.GroupNorm(32, channels[0])
        self.conv_out = nn.Conv2d(channels[0], out_channels, 3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.conv_in(x)

        # Middle
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        # Upsampling
        for i, block in enumerate(self.up_blocks):
            x = block(x)
            # Upsample after every 2 residual blocks (accounting for attention blocks)
            if i < len(self.up_blocks) - 2 and i % 3 == 2:
                x = self.upsample(x)

        # Final upsampling
        x = self.upsample(x)

        # Output
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x


class VAE(nn.Module):
    """Variational Autoencoder for FLUX.1"""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.encoder = Encoder(
            config.in_channels,
            config.channels,
            config.z_channels
        )
        self.decoder = Decoder(
            config.z_channels,
            config.channels,
            config.in_channels
        )
        self.quant_conv = nn.Conv2d(2 * config.z_channels, 2 * config.z_channels, 1)
        self.post_quant_conv = nn.Conv2d(config.z_channels, config.z_channels, 1)
        self.z_channels = config.z_channels

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mean, logvar


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""

    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        except:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)

        self.features = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = self.features(x)
        y_features = self.features(y)
        return F.mse_loss(x_features, y_features)


class VAELoss(nn.Module):
    """Combined VAE loss with reconstruction, KL divergence, and perceptual loss"""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.kl_weight = config.kl_weight
        self.perceptual_weight = config.perceptual_weight
        self.reconstruction_weight = config.reconstruction_weight
        self.perceptual_loss = PerceptualLoss()

    def forward(self, reconstruction, target, mean, logvar):
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, target, reduction='mean')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = kl_loss / (target.shape[0] * target.shape[2] * target.shape[3])

        # Perceptual loss
        perceptual = self.perceptual_loss(reconstruction, target)

        # Combined loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.kl_weight * kl_loss +
            self.perceptual_weight * perceptual
        )

        return total_loss, {
            'reconstruction_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'perceptual_loss': perceptual.item(),
            'total_loss': total_loss.item()
        }


class VAETrainer:
    """Trainer class for VAE"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.samples_dir = self.output_dir / 'samples'
        self.logs_dir = self.output_dir / 'logs'

        for dir_path in [self.checkpoint_dir, self.samples_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Save config
        config.save(self.output_dir / 'config.json')

        # Initialize model
        logger.info("Initializing VAE model...")
        self.model = VAE(config).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Loss and optimizer
        self.criterion = VAELoss(config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=1e-6
        )

        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None

        # TensorBoard
        self.writer = SummaryWriter(self.logs_dir)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Resume from checkpoint if specified
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

    def prepare_data(self):
        """Prepare data loaders"""
        logger.info("Preparing data loaders...")

        # Get all image paths
        data_path = Path(self.config.data_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_paths = [
            str(p) for p in data_path.rglob('*')
            if p.suffix.lower() in image_extensions
        ]

        logger.info(f"Found {len(image_paths)} images")

        if len(image_paths) == 0:
            raise ValueError(f"No images found in {data_path}")

        # Split into train and validation
        val_size = int(len(image_paths) * self.config.val_split)
        train_paths = image_paths[val_size:]
        val_paths = image_paths[:val_size]

        logger.info(f"Training samples: {len(train_paths)}")
        logger.info(f"Validation samples: {len(val_paths)}")

        # Data transforms
        transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.CenterCrop(self.config.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.CenterCrop(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Create datasets
        train_dataset = ImageDataset(train_paths, transform)
        val_dataset = ImageDataset(val_paths, val_transform)

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False
        )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0,
            'perceptual_loss': 0.0
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)

            # Forward pass
            if self.scaler:
                with autocast():
                    reconstruction, mean, logvar = self.model(images)
                    loss, loss_dict = self.criterion(reconstruction, images, mean, logvar)

                # Backward pass
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    if self.config.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                reconstruction, mean, logvar = self.model(images)
                loss, loss_dict = self.criterion(reconstruction, images, mean, logvar)

                loss.backward()

                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    if self.config.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Update metrics
            for key, value in loss_dict.items():
                epoch_losses[key] += value

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'recon': f"{loss_dict['reconstruction_loss']:.4f}",
                'kl': f"{loss_dict['kl_loss']:.6f}"
            })

            # Log to TensorBoard
            if self.global_step % 10 == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', value, self.global_step)
                self.writer.add_scalar(
                    'train/learning_rate',
                    self.optimizer.param_groups[0]['lr'],
                    self.global_step
                )

            # Save samples
            if self.global_step % self.config.save_samples_freq == 0:
                self.save_samples(images, reconstruction, self.global_step)

            self.global_step += 1

        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0,
            'perceptual_loss': 0.0
        }

        pbar = tqdm(self.val_loader, desc="Validation")
        for images, _ in pbar:
            images = images.to(self.device)

            reconstruction, mean, logvar = self.model(images)
            loss, loss_dict = self.criterion(reconstruction, images, mean, logvar)

            for key, value in loss_dict.items():
                val_losses[key] += value

            pbar.set_postfix({'loss': f"{loss_dict['total_loss']:.4f}"})

        # Average losses
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches

        # Log to TensorBoard
        for key, value in val_losses.items():
            self.writer.add_scalar(f'val/{key}', value, self.current_epoch)

        return val_losses

    def save_samples(self, original, reconstruction, step):
        """Save sample reconstructions"""
        n_samples = min(8, original.shape[0])
        comparison = torch.cat([
            original[:n_samples],
            reconstruction[:n_samples]
        ])

        # Denormalize
        comparison = comparison * 0.5 + 0.5
        comparison = torch.clamp(comparison, 0, 1)

        grid = make_grid(comparison, nrow=n_samples)
        save_image(grid, self.samples_dir / f'sample_step_{step}.png')

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logger.info(f"Resumed from epoch {self.current_epoch}")

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")

        # Prepare data
        self.prepare_data()

        # Training loop
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_losses = self.train_epoch()

            logger.info(
                f"Epoch {epoch} - "
                f"Train Loss: {train_losses['total_loss']:.4f}, "
                f"Recon: {train_losses['reconstruction_loss']:.4f}, "
                f"KL: {train_losses['kl_loss']:.6f}"
            )

            # Validate
            if epoch % self.config.val_freq == 0:
                val_losses = self.validate()
                logger.info(
                    f"Epoch {epoch} - "
                    f"Val Loss: {val_losses['total_loss']:.4f}, "
                    f"Recon: {val_losses['reconstruction_loss']:.4f}, "
                    f"KL: {val_losses['kl_loss']:.6f}"
                )

                # Save best model
                if val_losses['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total_loss']
                    self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', is_best=True)

            # Save checkpoint
            if epoch % self.config.checkpoint_freq == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

            # Update learning rate
            self.scheduler.step()

        logger.info("Training completed!")
        self.writer.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train FLUX.1 VAE')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training images')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')

    # Model
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent dimension')
    parser.add_argument('--z_channels', type=int, default=16,
                        help='Number of latent channels')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Input image size')

    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')

    # Loss weights
    parser.add_argument('--kl_weight', type=float, default=1e-6,
                        help='KL divergence weight')
    parser.add_argument('--perceptual_weight', type=float, default=1.0,
                        help='Perceptual loss weight')

    # Training settings
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping threshold')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Checkpointing
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_freq', type=int, default=1,
                        help='Checkpoint save frequency (epochs)')

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Create config from arguments
    config = TrainingConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        latent_dim=args.latent_dim,
        z_channels=args.z_channels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        kl_weight=args.kl_weight,
        perceptual_weight=args.perceptual_weight,
        mixed_precision=args.mixed_precision,
        gradient_clip=args.gradient_clip,
        num_workers=args.num_workers,
        resume_from=args.resume_from,
        checkpoint_freq=args.checkpoint_freq
    )

    # Create trainer and start training
    trainer = VAETrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
