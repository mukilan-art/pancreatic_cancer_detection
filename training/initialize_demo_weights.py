"""
Script to initialize baseline model architecture weights for U-Net and ResNet50.
This ensures the backend system loads both models on startup, enabling real predictions
and Grad-CAM explainability out of the box.
"""
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.segmentation.unet_model import build_unet
from backend.app.classification.resnet50_model import build_resnet50

def generate_weights():
    root_dir = Path(__file__).parent.parent
    unet_path = root_dir / "backend" / "models" / "unet" / "unet_pancreas.h5"
    resnet_path = root_dir / "backend" / "models" / "resnet50" / "resnet50_classifier.h5"

    unet_path.parent.mkdir(parents=True, exist_ok=True)
    resnet_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building and saving U-Net model to {unet_path}...")
    unet = build_unet(input_size=(512, 512, 1))
    unet.compile(optimizer='adam', loss='binary_crossentropy')
    unet.save(str(unet_path))

    print(f"Building and saving ResNet50 model to {resnet_path}...")
    resnet = build_resnet50(num_classes=2)
    resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    resnet.save(str(resnet_path))

    print("Model weights initialized successfully!")

if __name__ == "__main__":
    generate_weights()
