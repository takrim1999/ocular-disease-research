# ResNet Notebook Guide

## 🎯 **Which Notebook Should You Run?**

### **For Basic Classification (Recommended First):**
**File:** `resnet_classification_general.ipynb`
- **Purpose:** Basic ResNet50 classification on ODIR dataset
- **Dataset:** Only uses `/kaggle/input/ocular-disease-recognition-odir5k/`
- **What it does:** 
  - Loads ODIR dataset
  - Creates train/val/test splits
  - Trains ResNet50 model
  - Saves model as `resnet_ocular_disease_model.h5`

### **For Traditional Data Augmentation:**
**File:** `resnet_traditional_data_augmentation.ipynb`
- **Purpose:** Applies traditional augmentation + ResNet50
- **Dataset:** Uses `/kaggle/input/ocular-disease-recognition-odir5k/`
- **What it does:**
  - Loads ODIR dataset
  - Applies flipping, rotation, brightness, contrast
  - Trains ResNet50 on augmented data

### **For Synthetic Data Classification:**
**File:** `resnet_synthetic_data_classification.ipynb`
- **Purpose:** Classifies synthetic images
- **Dataset:** Requires pre-trained model + synthetic images
- **What it does:**
  - Loads trained ResNet50 model
  - Classifies synthetic images
  - Generates confusion matrix

### **For Combined Real + Synthetic Data:**
**File:** `resnet_synthetic_and_real_data_combined.ipynb`
- **Purpose:** Trains on both real and synthetic data
- **Dataset:** Requires `/kaggle/input/combined-tsne-new-1/combined_tsne_new-1.csv`
- **What it does:**
  - Loads real ODIR data
  - Loads synthetic t-SNE selected data
  - Combines datasets
  - Trains ResNet50 on combined data

## 🚨 **Error Explanation**

The error you encountered:
```
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/combined-tsne-new-1/combined_tsne_new-1.csv'
```

**This means you're running the WRONG notebook!** 

You're trying to run `resnet_synthetic_and_real_data_combined.ipynb` which requires synthetic data that doesn't exist in the basic ODIR dataset.

## ✅ **What You Should Do**

### **Option 1: Start with Basic Classification (Recommended)**
1. Run `resnet_classification_general.ipynb`
2. This only needs the ODIR dataset: `/kaggle/input/ocular-disease-recognition-odir5k/`
3. No synthetic data required

### **Option 2: Run Traditional Augmentation**
1. Run `resnet_traditional_data_augmentation.ipynb`
2. This also only needs the ODIR dataset
3. No synthetic data required

### **Option 3: Skip Synthetic Data for Now**
- The synthetic data notebooks require additional datasets that aren't available in the basic ODIR package
- Focus on the basic classification first

## 🔧 **Quick Fix**

If you want to run the basic classification right now:

1. **Make sure you're running:** `resnet_classification_general.ipynb`
2. **Check the dataset path:** Should be `/kaggle/input/ocular-disease-recognition-odir5k/`
3. **Verify the first cell** loads the ODIR dataset, not synthetic data

## 📋 **Expected Dataset Structure**

For basic classification, you should see:
```
/kaggle/input/ocular-disease-recognition-odir5k/
├── preprocessed_images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── full_df.csv
```

**NOT:**
```
/kaggle/input/combined-tsne-new-1/  ← This doesn't exist in basic ODIR
```
