#!/usr/bin/env python3
"""
Installation script for ResNet-based Ocular Disease Classification
This script installs the required dependencies for running the notebooks.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")
        return False

def main():
    print("Installing dependencies for ResNet-based Ocular Disease Classification...")
    print("=" * 70)
    
    # Core dependencies
    core_packages = [
        "tensorflow>=2.11.0",
        "tensorflow-addons>=0.19.0",
        "numpy>=1.22.4",
        "pandas>=1.5.3",
        "matplotlib>=3.6.3",
        "seaborn>=0.12.2",
        "opencv-python-headless>=4.7.0.72",
        "scikit-learn>=1.2.2",
        "Pillow>=9.0.0",
        "scipy>=1.9.0"
    ]
    
    print("\nInstalling core dependencies...")
    success_count = 0
    for package in core_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\nCore dependencies: {success_count}/{len(core_packages)} installed successfully")
    
    # Optional dependencies for advanced features
    optional_packages = [
        "torch>=1.13.1",
        "torchvision>=0.14.0",
        "diffusers[torch]>=0.10.2",
        "transformers>=4.28.1"
    ]
    
    print("\nInstalling optional dependencies...")
    optional_success = 0
    for package in optional_packages:
        if install_package(package):
            optional_success += 1
    
    print(f"\nOptional dependencies: {optional_success}/{len(optional_packages)} installed successfully")
    
    print("\n" + "=" * 70)
    print("Installation complete!")
    print("\nIf tensorflow-addons installation failed, the notebooks will use")
    print("alternative F1 score implementations.")
    print("\nTo run the notebooks:")
    print("1. Upload them to Kaggle or Google Colab")
    print("2. Use the ODIR dataset: https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k")
    print("3. Update the dataset paths in the notebooks if needed")

if __name__ == "__main__":
    main()
