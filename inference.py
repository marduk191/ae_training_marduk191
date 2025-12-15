#!/usr/bin/env python3
"""
FLUX.1 VAE Inference Script
Use trained VAE models for encoding and decoding images.

Requires Python 3.12+
"""

import argparse
import logging
from pathlib import Path
from typing import Union, List

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np

# Import from train_vae
from train_vae import VAE, TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VAEInference:
    """Inference wrapper for VAE model"""

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Initialize VAE for inference

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load config
        config_dict = checkpoint.get('config', {})
        self.config = TrainingConfig(**config_dict)

        # Initialize model
        self.model = VAE(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info("Model loaded successfully")

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.CenterCrop(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.inverse_transform = transforms.Compose([
            transforms.Normalize([-1, -1, -1], [2, 2, 2]),  # Denormalize
        ])

    @torch.no_grad()
    def encode_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Encode an image to latent representation

        Args:
            image: Path to image file or PIL Image

        Returns:
            Latent tensor
        """
        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Encode
        mean, logvar = self.model.encode(image_tensor)

        # Use mean as the latent (for deterministic encoding)
        # For stochastic encoding, use: latent = self.model.reparameterize(mean, logvar)
        latent = mean

        return latent

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """
        Decode a latent tensor to an image

        Args:
            latent: Latent tensor

        Returns:
            PIL Image
        """
        # Decode
        reconstruction = self.model.decode(latent)

        # Convert to PIL image
        image = self.tensor_to_image(reconstruction[0])

        return image

    @torch.no_grad()
    def reconstruct_image(self, image: Union[str, Image.Image]) -> Image.Image:
        """
        Reconstruct an image through encode-decode cycle

        Args:
            image: Path to image file or PIL Image

        Returns:
            Reconstructed PIL Image
        """
        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Encode and decode
        reconstruction, mean, logvar = self.model(image_tensor)

        # Convert to PIL image
        reconstructed = self.tensor_to_image(reconstruction[0])

        return reconstructed

    @torch.no_grad()
    def interpolate_between_images(
        self,
        image1: Union[str, Image.Image],
        image2: Union[str, Image.Image],
        steps: int = 10
    ) -> List[Image.Image]:
        """
        Interpolate between two images in latent space

        Args:
            image1: First image
            image2: Second image
            steps: Number of interpolation steps

        Returns:
            List of interpolated images
        """
        # Encode both images
        latent1 = self.encode_image(image1)
        latent2 = self.encode_image(image2)

        # Interpolate
        interpolated_images = []
        for i in range(steps):
            alpha = i / (steps - 1)
            latent = (1 - alpha) * latent1 + alpha * latent2

            # Decode
            image = self.decode_latent(latent)
            interpolated_images.append(image)

        return interpolated_images

    @torch.no_grad()
    def generate_variations(
        self,
        image: Union[str, Image.Image],
        num_variations: int = 5,
        temperature: float = 0.5
    ) -> List[Image.Image]:
        """
        Generate variations of an image by sampling from latent distribution

        Args:
            image: Input image
            num_variations: Number of variations to generate
            temperature: Sampling temperature (higher = more variation)

        Returns:
            List of variation images
        """
        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Encode
        mean, logvar = self.model.encode(image_tensor)

        # Generate variations
        variations = []
        for _ in range(num_variations):
            # Sample from latent distribution with temperature
            std = torch.exp(0.5 * logvar) * temperature
            eps = torch.randn_like(std)
            latent = mean + eps * std

            # Decode
            reconstruction = self.model.decode(latent)
            variation = self.tensor_to_image(reconstruction[0])
            variations.append(variation)

        return variations

    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image"""
        # Denormalize
        tensor = tensor * 0.5 + 0.5
        tensor = torch.clamp(tensor, 0, 1)

        # Convert to numpy
        image_np = tensor.cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)

        return Image.fromarray(image_np)

    @torch.no_grad()
    def batch_reconstruct(
        self,
        image_paths: List[str],
        output_dir: str,
        save_comparison: bool = True
    ):
        """
        Reconstruct multiple images and save results

        Args:
            image_paths: List of image paths
            output_dir: Output directory for reconstructions
            save_comparison: If True, save original vs reconstruction comparison
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {len(image_paths)} images...")

        for image_path in image_paths:
            # Load image
            image = Image.open(image_path).convert('RGB')
            filename = Path(image_path).stem

            # Reconstruct
            reconstructed = self.reconstruct_image(image)

            if save_comparison:
                # Create side-by-side comparison
                comparison = Image.new('RGB', (image.width * 2, image.height))
                comparison.paste(image, (0, 0))
                comparison.paste(reconstructed, (image.width, 0))
                comparison.save(output_path / f'{filename}_comparison.png')
            else:
                reconstructed.save(output_path / f'{filename}_reconstructed.png')

        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='VAE Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['reconstruct', 'encode', 'interpolate', 'variations'],
                        help='Inference mode')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path(s)')
    parser.add_argument('--input2', type=str, default=None,
                        help='Second input image (for interpolation)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path or directory')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of interpolation steps')
    parser.add_argument('--num_variations', type=int, default=5,
                        help='Number of variations to generate')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Sampling temperature for variations')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    # Initialize inference
    vae = VAEInference(args.checkpoint, args.device)

    # Execute based on mode
    if args.mode == 'reconstruct':
        logger.info(f"Reconstructing image: {args.input}")
        reconstructed = vae.reconstruct_image(args.input)
        reconstructed.save(args.output)
        logger.info(f"Saved to {args.output}")

    elif args.mode == 'encode':
        logger.info(f"Encoding image: {args.input}")
        latent = vae.encode_image(args.input)
        torch.save(latent, args.output)
        logger.info(f"Latent saved to {args.output}")
        logger.info(f"Latent shape: {latent.shape}")

    elif args.mode == 'interpolate':
        if not args.input2:
            raise ValueError("--input2 required for interpolation mode")

        logger.info(f"Interpolating between {args.input} and {args.input2}")
        interpolations = vae.interpolate_between_images(
            args.input, args.input2, args.steps
        )

        # Save interpolations
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(interpolations):
            img.save(output_dir / f'interpolation_{i:03d}.png')

        logger.info(f"Saved {len(interpolations)} interpolations to {args.output}")

    elif args.mode == 'variations':
        logger.info(f"Generating variations of {args.input}")
        variations = vae.generate_variations(
            args.input, args.num_variations, args.temperature
        )

        # Save variations
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(variations):
            img.save(output_dir / f'variation_{i:03d}.png')

        logger.info(f"Saved {len(variations)} variations to {args.output}")


if __name__ == '__main__':
    main()
