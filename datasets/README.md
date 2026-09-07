# Dataset Setup Guide

## TCIA Pancreas-CT Dataset

This project uses the Pancreas-CT dataset from The Cancer Imaging Archive (TCIA).

## Download Instructions

1. Visit: https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT
2. Create a TCIA account if required
3. Download the TCIA client
4. Download the Pancreas-CT dataset (approximately 75GB)
5. Place the DICOM files in `datasets/pancreas-ct/`

## Expected Directory Structure

### For U-Net Segmentation Training:
```
datasets/
    pancreas-ct/
        images/
            case_0001_slice_0001.png
            case_0001_slice_0002.png
            ...
        masks/
            case_0001_slice_0001.png  (binary masks: 0=background, 255=pancreas)
            case_0001_slice_0002.png
            ...
```

### For ResNet50 Classification Training:
```
datasets/
    classification/
        Normal/
            normal_001.png
            normal_002.png
            ...
        Pancreatic Cancer/
            cancer_001.png
            cancer_002.png
            ...
```

### After Preprocessing:
```
datasets/
    processed/
        train/
            images/
            masks/
        val/
            images/
            masks/
        test/
            images/
            masks/
```

## Preprocessing Steps

1. Convert DICOM to PNG slices
2. Apply CLAHE enhancement
3. Resize to 512x512 (segmentation) or 224x224 (classification)
4. Train/val/test split (70/15/15)

## Privacy and Compliance

- NEVER commit patient data to Git
- datasets/ is in .gitignore
- Handle data in accordance with applicable regulations

## Running Preprocessing

```bash
cd /path/to/project
python training/preprocess_dataset.py --data_dir datasets/pancreas-ct --output_dir datasets/processed
```
