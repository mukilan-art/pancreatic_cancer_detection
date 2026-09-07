"""
Dataset Preprocessing Script
Preprocesses the TCIA Pancreas-CT dataset for training.

Usage:
    python training/preprocess_dataset.py --data_dir datasets/pancreas-ct --output_dir datasets/processed

Expected input structure:
    datasets/pancreas-ct/
        images/  (CT slices as NIfTI .nii or DICOM or PNG)
        masks/   (corresponding segmentation masks)
"""
import argparse
import logging
import sys
import os
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from backend.app.preprocessing.clahe_processor import CLAHEProcessor
except ImportError:
    class CLAHEProcessor:
        def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
            self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        def apply(self, img):
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return self.clahe.apply(img)

from training.config import unet_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess TCIA Pancreas-CT dataset")
    parser.add_argument("--data_dir", type=str, default=unet_config.data_dir, help="Input dataset directory")
    parser.add_argument("--output_dir", type=str, default=str(Path(unet_config.data_dir).parent / "processed"), help="Output processed directory")
    return parser.parse_args()

def load_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in ['.png', '.jpg', '.jpeg']:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    elif suffix in ['.nii', '.gz']:
        try:
            import nibabel as nib
            img = nib.load(str(path)).get_fdata()
            # Handle NIfTI volumes by taking middle slice for simplicity in this script
            if len(img.shape) >= 3:
                img = img[:, :, img.shape[2] // 2]
            return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8) * 255.0
        except ImportError:
            logging.error("nibabel is required for NIfTI files.")
            return None
    elif suffix in ['.dcm']:
        try:
            import pydicom
            ds = pydicom.dcmread(str(path))
            return ds.pixel_array
        except ImportError:
            logging.error("pydicom is required for DICOM files.")
            return None
    else:
        logging.warning(f"Unsupported file format: {suffix}")
        return None

def preprocess_and_save(img_path, mask_path, output_img_dir, output_mask_dir, clahe_proc):
    img = load_image(img_path)
    mask = load_image(mask_path)
    
    if img is None or mask is None:
        return False
        
    img = img.astype(np.uint8)
    mask = mask.astype(np.uint8)
    
    # Apply CLAHE
    img = clahe_proc.apply(img)
    
    # Resize
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Binarize mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    out_img_path = output_img_dir / (img_path.stem + ".png")
    out_mask_path = output_mask_dir / (mask_path.stem + ".png")
    
    cv2.imwrite(str(out_img_path), img)
    cv2.imwrite(str(out_mask_path), mask)
    
    return True

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    
    if not img_dir.exists() or not mask_dir.exists():
        logging.error(f"Input directories not found: {img_dir} or {mask_dir}")
        return
        
    train_img_dir = output_dir / "train" / "images"
    train_mask_dir = output_dir / "train" / "masks"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_mask_dir.mkdir(parents=True, exist_ok=True)
    
    clahe_proc = CLAHEProcessor()
    
    img_files = sorted(list(img_dir.glob("*.*")))
    mask_files = sorted(list(mask_dir.glob("*.*")))
    
    # Basic alignment matching names
    processed_count = 0
    for img_path in img_files:
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            if preprocess_and_save(img_path, mask_path, train_img_dir, train_mask_dir, clahe_proc):
                processed_count += 1
                if processed_count % 100 == 0:
                    logging.info(f"Processed {processed_count} files...")
                    
    logging.info(f"Finished processing {processed_count} files.")

if __name__ == "__main__":
    main()
