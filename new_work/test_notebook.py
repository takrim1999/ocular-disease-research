#!/usr/bin/env python3
"""
Test script to verify the ResNet classification notebook works correctly
This script tests the key components without running the full training
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    try:
        import cv2
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_tensorflow_addons():
    """Test tensorflow_addons availability"""
    print("Testing tensorflow_addons...")
    try:
        import tensorflow_addons as tfa
        print("✅ tensorflow_addons available")
        return True
    except ImportError:
        print("⚠️ tensorflow_addons not available - will use fallback")
        return False

def test_gpu():
    """Test GPU availability"""
    print("Testing GPU...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU available: {len(gpus)} device(s)")
        return True
    else:
        print("⚠️ No GPU available - will use CPU")
        return False

def test_resnet_model():
    """Test ResNet50 model creation"""
    print("Testing ResNet50 model creation...")
    try:
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Dense
        from tensorflow.keras import Sequential
        
        # Create a small test model
        resnet = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        model = Sequential()
        model.add(resnet)
        model.add(Dropout(0.5))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(5, activation="softmax"))
        
        print("✅ ResNet50 model created successfully")
        print(f"Model parameters: {model.count_params():,}")
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_dataset_paths():
    """Test if dataset paths are correctly configured"""
    print("Testing dataset paths...")
    
    # Test Kaggle paths
    kaggle_path = "/kaggle/input/ocular-disease-recognition-odir5k/"
    if os.path.exists(kaggle_path):
        print("✅ Kaggle dataset path exists")
        return True
    else:
        print("⚠️ Kaggle dataset path not found (expected if not running on Kaggle)")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing ResNet Classification Notebook Components")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_tensorflow_addons,
        test_gpu,
        test_resnet_model,
        test_dataset_paths
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("=" * 60)
    print("📊 Test Results Summary:")
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Notebook should work correctly.")
    else:
        print("⚠️ Some tests failed. Check the issues above.")
    
    print("\n💡 To run the full notebook:")
    print("1. Upload to Kaggle and add ODIR dataset")
    print("2. Or run locally with the ODIR dataset")
    print("3. Enable GPU for best performance")

if __name__ == "__main__":
    main()
