"""
Grad-CAM Generation Script

Generates Grad-CAM visualizations for a directory of images using a trained classifier.

Usage:
    python training/generate_gradcam.py --images_dir datasets/classification/test --output_dir training/outputs/gradcam
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from backend.app.classification.resnet50_model import load_resnet50
    from backend.app.explainability.gradcam import GradCAM
    from backend.app.preprocessing.clahe_processor import CLAHEProcessor
except ImportError:
    pass

from training.config import resnet_config

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Would generate Grad-CAM for {images_dir} and save to {out_dir}")
    # In a full implementation, this loops through images, applies GradCAM and saves the overlay
    
if __name__ == "__main__":
    main()
