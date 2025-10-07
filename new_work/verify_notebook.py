#!/usr/bin/env python3
"""
Script to verify which notebook you should be running
and check dataset availability
"""

import os

def check_dataset_paths():
    """Check which datasets are available"""
    print("🔍 Checking available datasets...")
    
    # Check basic ODIR dataset
    odir_path = "/kaggle/input/ocular-disease-recognition-odir5k/"
    if os.path.exists(odir_path):
        print("✅ ODIR dataset found: /kaggle/input/ocular-disease-recognition-odir5k/")
        
        # Check specific files
        csv_path = os.path.join(odir_path, "full_df.csv")
        img_path = os.path.join(odir_path, "preprocessed_images")
        
        if os.path.exists(csv_path):
            print("  ✅ full_df.csv found")
        else:
            print("  ❌ full_df.csv not found")
            
        if os.path.exists(img_path):
            print("  ✅ preprocessed_images directory found")
        else:
            print("  ❌ preprocessed_images directory not found")
    else:
        print("❌ ODIR dataset not found: /kaggle/input/ocular-disease-recognition-odir5k/")
    
    # Check synthetic dataset
    synthetic_path = "/kaggle/input/combined-tsne-new-1/"
    if os.path.exists(synthetic_path):
        print("✅ Synthetic dataset found: /kaggle/input/combined-tsne-new-1/")
    else:
        print("❌ Synthetic dataset not found: /kaggle/input/combined-tsne-new-1/")
    
    print()

def recommend_notebook():
    """Recommend which notebook to run based on available datasets"""
    print("📋 Notebook Recommendations:")
    
    odir_available = os.path.exists("/kaggle/input/ocular-disease-recognition-odir5k/")
    synthetic_available = os.path.exists("/kaggle/input/combined-tsne-new-1/")
    
    if odir_available and not synthetic_available:
        print("🎯 RECOMMENDED: Run resnet_classification_general.ipynb")
        print("   - Uses only ODIR dataset")
        print("   - No synthetic data required")
        print("   - Perfect for basic classification")
        
    elif odir_available and synthetic_available:
        print("🎯 AVAILABLE OPTIONS:")
        print("   1. resnet_classification_general.ipynb (basic classification)")
        print("   2. resnet_traditional_data_augmentation.ipynb (with augmentation)")
        print("   3. resnet_synthetic_and_real_data_combined.ipynb (with synthetic data)")
        
    elif not odir_available:
        print("❌ Cannot run any notebooks - ODIR dataset not found!")
        print("   Please add the ODIR dataset to your Kaggle notebook")
        
    print()

def check_current_cell():
    """Check if the current cell is trying to load synthetic data"""
    print("🔍 Current Cell Analysis:")
    print("If you see this error:")
    print("   FileNotFoundError: '/kaggle/input/combined-tsne-new-1/combined_tsne_new-1.csv'")
    print()
    print("It means you're running the WRONG notebook!")
    print("You should be running: resnet_classification_general.ipynb")
    print("NOT: resnet_synthetic_and_real_data_combined.ipynb")
    print()

def main():
    print("🔧 ResNet Notebook Verification Tool")
    print("=" * 50)
    
    check_dataset_paths()
    recommend_notebook()
    check_current_cell()
    
    print("💡 Next Steps:")
    print("1. Make sure you're running the correct notebook")
    print("2. Verify the dataset is properly added")
    print("3. Check the first cell loads the ODIR dataset")
    print("4. Run cells sequentially from top to bottom")

if __name__ == "__main__":
    main()
