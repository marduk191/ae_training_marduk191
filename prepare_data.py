#!/usr/bin/env python3
"""
Data Preparation Utilities for VAE Training
Includes image preprocessing, dataset validation, and augmentation.

Requires Python 3.12+
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validate and clean image datasets"""

    def __init__(self, min_size: int = 256, max_size: int = 4096):
        self.min_size = min_size
        self.max_size = max_size
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def validate_image(self, image_path: Path) -> Tuple[bool, str]:
        """
        Validate a single image

        Returns:
            (is_valid, reason)
        """
        try:
            # Check file extension
            if image_path.suffix.lower() not in self.supported_formats:
                return False, f"Unsupported format: {image_path.suffix}"

            # Try to open image
            with Image.open(image_path) as img:
                # Check mode
                if img.mode not in ['RGB', 'L', 'RGBA']:
                    return False, f"Unsupported mode: {img.mode}"

                # Check size
                width, height = img.size
                if width < self.min_size or height < self.min_size:
                    return False, f"Image too small: {width}x{height}"

                if width > self.max_size or height > self.max_size:
                    return False, f"Image too large: {width}x{height}"

                # Check if image is corrupted
                img.verify()

            return True, "OK"

        except Exception as e:
            return False, f"Error: {str(e)}"

    def validate_dataset(
        self,
        data_dir: Path,
        output_file: str = None,
        num_workers: int = 4
    ) -> Tuple[List[Path], List[Tuple[Path, str]]]:
        """
        Validate all images in a directory

        Returns:
            (valid_images, invalid_images_with_reasons)
        """
        logger.info(f"Scanning directory: {data_dir}")

        # Find all image files
        all_files = [
            p for p in data_dir.rglob('*')
            if p.is_file() and p.suffix.lower() in self.supported_formats
        ]

        logger.info(f"Found {len(all_files)} potential image files")

        valid_images = []
        invalid_images = []

        # Validate images in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.validate_image, img_path): img_path
                for img_path in all_files
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Validating"):
                img_path = futures[future]
                is_valid, reason = future.result()

                if is_valid:
                    valid_images.append(img_path)
                else:
                    invalid_images.append((img_path, reason))
                    logger.warning(f"Invalid: {img_path} - {reason}")

        # Log summary
        logger.info(f"Valid images: {len(valid_images)}")
        logger.info(f"Invalid images: {len(invalid_images)}")

        # Save results if requested
        if output_file:
            results = {
                'total': len(all_files),
                'valid': len(valid_images),
                'invalid': len(invalid_images),
                'valid_images': [str(p) for p in valid_images],
                'invalid_images': [
                    {'path': str(p), 'reason': reason}
                    for p, reason in invalid_images
                ]
            }

            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"Results saved to {output_file}")

        return valid_images, invalid_images


class ImagePreprocessor:
    """Preprocess images for training"""

    def __init__(
        self,
        target_size: int = 512,
        quality: int = 95,
        convert_to_rgb: bool = True
    ):
        self.target_size = target_size
        self.quality = quality
        self.convert_to_rgb = convert_to_rgb

    def preprocess_image(
        self,
        image_path: Path,
        output_path: Path,
        resize_mode: str = 'cover'
    ) -> bool:
        """
        Preprocess a single image

        Args:
            image_path: Input image path
            output_path: Output image path
            resize_mode: 'cover' (crop to square) or 'contain' (pad to square)

        Returns:
            Success status
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if self.convert_to_rgb and img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize
                if resize_mode == 'cover':
                    # Crop to square (center crop)
                    width, height = img.size
                    min_dim = min(width, height)

                    left = (width - min_dim) // 2
                    top = (height - min_dim) // 2
                    right = left + min_dim
                    bottom = top + min_dim

                    img = img.crop((left, top, right, bottom))
                    img = img.resize((self.target_size, self.target_size), Image.LANCZOS)

                elif resize_mode == 'contain':
                    # Pad to square
                    width, height = img.size
                    max_dim = max(width, height)

                    # Create square canvas
                    new_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))

                    # Paste image in center
                    left = (max_dim - width) // 2
                    top = (max_dim - height) // 2
                    new_img.paste(img, (left, top))

                    img = new_img.resize((self.target_size, self.target_size), Image.LANCZOS)

                # Save
                output_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(output_path, quality=self.quality, optimize=True)

            return True

        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {e}")
            return False

    def preprocess_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        resize_mode: str = 'cover',
        num_workers: int = 4
    ) -> Tuple[int, int]:
        """
        Preprocess all images in a directory

        Returns:
            (num_processed, num_failed)
        """
        # Find all images
        image_files = [
            p for p in input_dir.rglob('*')
            if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        ]

        logger.info(f"Found {len(image_files)} images to preprocess")

        num_processed = 0
        num_failed = 0

        # Process images in parallel
        def process_image(img_path):
            # Maintain directory structure
            rel_path = img_path.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix('.jpg')

            success = self.preprocess_image(img_path, output_path, resize_mode)
            return success

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_image, img) for img in image_files]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                if future.result():
                    num_processed += 1
                else:
                    num_failed += 1

        logger.info(f"Processed: {num_processed}, Failed: {num_failed}")

        return num_processed, num_failed


class DatasetAnalyzer:
    """Analyze dataset statistics"""

    def __init__(self):
        self.stats = {
            'num_images': 0,
            'resolutions': [],
            'aspect_ratios': [],
            'mean_rgb': [0, 0, 0],
            'std_rgb': [0, 0, 0]
        }

    def analyze_image(self, image_path: Path) -> dict:
        """Analyze a single image"""
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size
            aspect_ratio = width / height

            # Calculate mean and std
            img_array = np.array(img) / 255.0
            mean = img_array.mean(axis=(0, 1))
            std = img_array.std(axis=(0, 1))

            return {
                'resolution': (width, height),
                'aspect_ratio': aspect_ratio,
                'mean': mean,
                'std': std
            }

    def analyze_dataset(self, data_dir: Path, num_samples: int = 1000) -> dict:
        """Analyze dataset statistics"""
        # Find all images
        image_files = [
            p for p in data_dir.rglob('*')
            if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        ]

        logger.info(f"Analyzing {len(image_files)} images...")

        # Sample if too many images
        if len(image_files) > num_samples:
            import random
            image_files = random.sample(image_files, num_samples)
            logger.info(f"Sampling {num_samples} images for analysis")

        resolutions = []
        aspect_ratios = []
        means = []
        stds = []

        for img_path in tqdm(image_files, desc="Analyzing"):
            try:
                stats = self.analyze_image(img_path)
                resolutions.append(stats['resolution'])
                aspect_ratios.append(stats['aspect_ratio'])
                means.append(stats['mean'])
                stds.append(stats['std'])
            except Exception as e:
                logger.warning(f"Error analyzing {img_path}: {e}")

        # Calculate statistics
        resolutions_array = np.array(resolutions)
        aspect_ratios_array = np.array(aspect_ratios)
        means_array = np.array(means)
        stds_array = np.array(stds)

        results = {
            'num_images': len(image_files),
            'resolution_stats': {
                'min_width': int(resolutions_array[:, 0].min()),
                'max_width': int(resolutions_array[:, 0].max()),
                'mean_width': float(resolutions_array[:, 0].mean()),
                'min_height': int(resolutions_array[:, 1].min()),
                'max_height': int(resolutions_array[:, 1].max()),
                'mean_height': float(resolutions_array[:, 1].mean()),
            },
            'aspect_ratio_stats': {
                'min': float(aspect_ratios_array.min()),
                'max': float(aspect_ratios_array.max()),
                'mean': float(aspect_ratios_array.mean()),
                'median': float(np.median(aspect_ratios_array)),
            },
            'color_stats': {
                'mean_rgb': means_array.mean(axis=0).tolist(),
                'std_rgb': stds_array.mean(axis=0).tolist(),
            }
        }

        return results


def main():
    parser = argparse.ArgumentParser(description='Data Preparation Utilities')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['validate', 'preprocess', 'analyze'],
                        help='Operation mode')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (for preprocess mode)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--target_size', type=int, default=512,
                        help='Target image size')
    parser.add_argument('--resize_mode', type=str, default='cover',
                        choices=['cover', 'contain'],
                        help='Resize mode: cover (crop) or contain (pad)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if args.mode == 'validate':
        validator = DatasetValidator()
        valid_images, invalid_images = validator.validate_dataset(
            input_dir,
            args.output_file,
            args.num_workers
        )
        logger.info(f"Validation complete: {len(valid_images)} valid, {len(invalid_images)} invalid")

    elif args.mode == 'preprocess':
        if not args.output_dir:
            raise ValueError("--output_dir required for preprocess mode")

        output_dir = Path(args.output_dir)
        preprocessor = ImagePreprocessor(target_size=args.target_size)
        num_processed, num_failed = preprocessor.preprocess_dataset(
            input_dir,
            output_dir,
            args.resize_mode,
            args.num_workers
        )
        logger.info(f"Preprocessing complete: {num_processed} processed, {num_failed} failed")

    elif args.mode == 'analyze':
        analyzer = DatasetAnalyzer()
        stats = analyzer.analyze_dataset(input_dir)

        # Print results
        print("\nDataset Analysis Results:")
        print("=" * 50)
        print(json.dumps(stats, indent=2))

        # Save if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")


if __name__ == '__main__':
    main()
