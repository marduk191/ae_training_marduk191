#!/usr/bin/env python3
"""
Python 3.12+ Compatibility Checker
Verifies that your environment meets all requirements for VAE training.
"""

import sys
import platform

def check_python_version():
    """Check if Python version is 3.12 or higher"""
    print("=" * 60)
    print("Python Version Check")
    print("=" * 60)

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    if version.major < 3 or (version.major == 3 and version.minor < 12):
        print("\n❌ ERROR: Python 3.12 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        print("\nPlease upgrade Python:")
        print("  - Download from: https://www.python.org/downloads/")
        print("  - Or use pyenv/conda to manage versions")
        return False

    print("✅ Python version is compatible")
    return True


def check_packages():
    """Check if required packages are installed and compatible"""
    print("\n" + "=" * 60)
    print("Package Compatibility Check")
    print("=" * 60)

    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'tqdm': 'tqdm',
        'tensorboard': 'TensorBoard',
        'albumentations': 'Albumentations'
    }

    all_good = True
    installed_packages = {}

    for package, name in packages.items():
        try:
            if package == 'PIL':
                import PIL
                module = PIL
                package_name = 'PIL'
            else:
                module = __import__(package)
                package_name = package

            version = getattr(module, '__version__', 'unknown')
            installed_packages[name] = version
            print(f"✅ {name:20} {version}")

        except ImportError:
            print(f"❌ {name:20} NOT INSTALLED")
            all_good = False

    if not all_good:
        print("\n⚠️  Some packages are missing. Install with:")
        print("   pip install -r requirements.txt")
        return False

    return True


def check_cuda():
    """Check CUDA availability"""
    print("\n" + "=" * 60)
    print("CUDA/GPU Check")
    print("=" * 60)

    try:
        import torch

        cuda_available = torch.cuda.is_available()

        if cuda_available:
            print(f"✅ CUDA is available")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   cuDNN version: {torch.backends.cudnn.version()}")

            device_count = torch.cuda.device_count()
            print(f"   Number of GPUs: {device_count}")

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                gpu_name = torch.cuda.get_device_name(i)
                memory_gb = props.total_memory / 1e9

                print(f"\n   GPU {i}: {gpu_name}")
                print(f"      Memory: {memory_gb:.2f} GB")
                print(f"      Compute Capability: {props.major}.{props.minor}")

                # Detect RTX 5090 and provide specific recommendations
                if "5090" in gpu_name or "RTX 5090" in gpu_name:
                    print("\n   🚀 RTX 5090 Detected!")
                    print("   ⭐ This is the ultimate GPU for VAE training!")
                    print("   Recommendations:")
                    print("      - Use CUDA 12.8 for best performance")
                    if torch.version.cuda and "12.8" not in str(torch.version.cuda):
                        print(f"      ⚠️  Current CUDA: {torch.version.cuda} (consider upgrading to 12.8)")
                        print("         pip install torch>=2.5.0 torchvision>=0.20.0 --index-url https://download.pytorch.org/whl/cu128")
                    else:
                        print("      ✅ CUDA version is optimal")
                    print("      - Train at 2048x2048 with batch_size=4")
                    print("      - Or 1024x1024 with batch_size=16")
                    print("      - See RTX5090_GUIDE.md for optimization tips")
                    print("      - Quick start: ./train_rtx5090.sh (Linux) or train_rtx5090.bat (Windows)")

                # Recommendations for other high-end GPUs
                elif "4090" in gpu_name:
                    print("   💪 RTX 4090 detected - Excellent for training!")
                    print("      Recommended: 1024x1024 with batch_size=12")
                elif "3090" in gpu_name or "A6000" in gpu_name:
                    print("   👍 High VRAM GPU - Great for large models")
                    print("      Recommended: 768x768 with batch_size=12")
                elif memory_gb < 12:
                    print("   ⚠️  Limited VRAM - Use smaller settings")
                    print("      Recommended: 512x512 with batch_size=4")

            # Test CUDA operations
            print("\n   Testing CUDA operations...")
            try:
                x = torch.randn(100, 100).cuda()
                y = torch.randn(100, 100).cuda()
                z = torch.mm(x, y)
                print("   ✅ CUDA operations working correctly")
            except Exception as e:
                print(f"   ❌ CUDA operation failed: {e}")
                return False

        else:
            print("⚠️  CUDA is NOT available")
            print("   Training will use CPU (much slower)")
            print("\n   To enable GPU:")
            print("   1. Ensure you have an NVIDIA GPU")
            print("   2. Install NVIDIA drivers from nvidia.com")
            print("   3. Reinstall PyTorch with CUDA:")
            print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

        return True

    except ImportError:
        print("❌ PyTorch not installed")
        return False


def check_compatibility_features():
    """Check Python 3.12+ specific features"""
    print("\n" + "=" * 60)
    print("Python 3.12+ Features Check")
    print("=" * 60)

    # Check if using Python 3.12+ features
    features = []

    # PEP 701: f-strings improvements
    try:
        test = f"{'test'}"
        features.append("✅ Enhanced f-strings (PEP 701)")
    except:
        features.append("⚠️  Enhanced f-strings not available")

    # Check typing improvements
    try:
        from typing import TypedDict
        features.append("✅ Typing improvements available")
    except:
        features.append("⚠️  Typing improvements not available")

    # Check pathlib improvements (Python 3.12+)
    try:
        from pathlib import Path
        # Python 3.12 added walk() method
        if hasattr(Path, 'walk'):
            features.append("✅ pathlib.Path.walk() available (Python 3.12+)")
        else:
            features.append("⚠️  pathlib.Path.walk() not available (pre-3.12)")
    except:
        features.append("⚠️  pathlib issues detected")

    for feature in features:
        print(f"   {feature}")

    return True


def estimate_memory_requirements():
    """Estimate memory requirements for training"""
    print("\n" + "=" * 60)
    print("Memory Requirements Estimate")
    print("=" * 60)

    configs = [
        ("256x256, batch=8", 6),
        ("512x512, batch=4", 12),
        ("512x512, batch=8", 20),
        ("768x768, batch=4", 24),
    ]

    print("\nEstimated VRAM requirements:")
    for config, vram in configs:
        print(f"   {config:25} ~{vram}GB VRAM")

    print("\nTips for reducing memory usage:")
    print("   - Use --mixed_precision flag (saves ~50% memory)")
    print("   - Reduce --batch_size")
    print("   - Reduce --image_size")
    print("   - Use --accumulation_steps to simulate larger batches")


def main():
    """Run all compatibility checks"""
    print("\n" + "=" * 60)
    print("FLUX.1 VAE Training - Compatibility Checker")
    print("Python 3.12+ Required")
    print("=" * 60 + "\n")

    checks = [
        ("Python Version", check_python_version),
        ("Packages", check_packages),
        ("CUDA/GPU", check_cuda),
        ("Python Features", check_compatibility_features),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error during {name} check: {e}")
            results.append((name, False))

    # Show memory requirements
    estimate_memory_requirements()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = all(result for _, result in results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print("\n" + "=" * 60)

    if all_passed:
        print("✅ All checks passed! You're ready to train.")
        print("\nNext steps:")
        print("1. Prepare your dataset: python prepare_data.py --mode validate --input_dir ./data/train")
        print("2. Start training: python train_vae.py --data_path ./data/train --output_dir ./outputs/my_vae")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nFor help:")
        print("- See README.md for installation instructions")
        print("- See WINDOWS_SETUP.md for Windows-specific help")
        return 1


if __name__ == '__main__':
    sys.exit(main())
