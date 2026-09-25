"""
Training configuration for the Pancreatic Cancer XAI project.
Modify these values before starting training.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATASET_DIR = ROOT_DIR / "datasets"
MODELS_DIR = ROOT_DIR / "backend" / "models"
OUTPUTS_DIR = ROOT_DIR / "training" / "outputs"

@dataclass
class UNetConfig:
    input_size: tuple = (512, 512, 1)
    batch_size: int = 1
    epochs: int = 100
    learning_rate: float = 1e-5
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    augmentation: bool = True
    checkpoint_path: str = str(MODELS_DIR / "unet" / "unet_pancreas.h5")
    data_dir: str = str(DATASET_DIR / "processed_tcia")
    images_subdir: str = "images"
    masks_subdir: str = "masks"

@dataclass
class ResNet50Config:
    input_size: tuple = (224, 224, 3)
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 1e-4
    fine_tune_lr: float = 1e-5
    validation_split: float = 0.15
    test_split: float = 0.15
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    augmentation: bool = True
    checkpoint_path: str = str(MODELS_DIR / "resnet50" / "resnet50_classifier.h5")
    data_dir: str = str(DATASET_DIR / "classification")
    classes: list = field(default_factory=lambda: ["Normal", "Pancreatic Cancer"])
    freeze_base_epochs: int = 10  # epochs to train with frozen base before fine-tuning

unet_config = UNetConfig()
resnet_config = ResNet50Config()
