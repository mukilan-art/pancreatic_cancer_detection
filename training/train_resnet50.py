"""
ResNet50 Transfer Learning Training Script for Pancreatic Cancer Classification

Usage:
    python training/train_resnet50.py
    python training/train_resnet50.py --epochs 30 --batch_size 8

Dataset structure expected:
    datasets/classification/
        Normal/           (CT images of normal pancreas)
        Pancreatic Cancer/  (CT images showing cancer)
"""
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:
    from backend.app.classification.resnet50_model import build_resnet50
except ImportError:
    def build_resnet50(input_shape, num_classes, freeze_base=True):
        base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = not freeze_base
        x = keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = keras.layers.Dense(256, activation='relu')(x)
        outputs = keras.layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')(x)
        return keras.Model(base_model.input, outputs)

from training.config import ResNet50Config, resnet_config

logging.basicConfig(level=logging.INFO)

class ClassificationDataGenerator(keras.utils.Sequence):
    def __init__(self, img_paths, labels, batch_size, img_size, augment=False):
        self.img_paths = img_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.img_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_paths = self.img_paths[i : i + self.batch_size]
        batch_labels = self.labels[i : i + self.batch_size]
        
        x = np.zeros((len(batch_paths), *self.img_size), dtype="float32")
        y = np.array(batch_labels, dtype="float32")
        
        for j, p in enumerate(batch_paths):
            img = cv2.imread(str(p))
            if img is not None:
                img = cv2.resize(img, self.img_size[:2])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if self.augment:
                    if np.random.rand() > 0.5:
                        img = np.fliplr(img)
                        
                x[j] = img / 255.0
                
        return x, y

def load_dataset(data_dir, classes):
    paths = []
    labels = []
    data_dir = Path(data_dir)
    for i, cls in enumerate(classes):
        cls_dir = data_dir / cls
        if cls_dir.exists():
            for p in cls_dir.glob("*.*"):
                if p.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    paths.append(p)
                    labels.append(i)
    return np.array(paths), np.array(labels)

def get_callbacks(config: ResNet50Config, phase: str):
    checkpoint_dir = Path(config.checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_finetune" if phase == 'finetune' else "_initial"
    return [
        keras.callbacks.ModelCheckpoint(str(checkpoint_dir / f"resnet_{phase}.h5"), save_best_only=True),
        keras.callbacks.EarlyStopping(patience=config.early_stopping_patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=config.reduce_lr_patience, factor=0.5, min_lr=1e-7),
    ]

def phase1_train(model, train_gen, val_gen, config, class_weight):
    logging.info("Phase 1: Training top layers (base frozen)")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
                  loss='binary_crossentropy' if len(config.classes) <= 2 else 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    callbacks = get_callbacks(config, 'initial')
    return model.fit(train_gen, validation_data=val_gen, epochs=config.freeze_base_epochs,
                     class_weight=class_weight, callbacks=callbacks)

def phase2_finetune(model, train_gen, val_gen, config, class_weight):
    logging.info("Phase 2: Fine-tuning full model")
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.fine_tune_lr),
                  loss='binary_crossentropy' if len(config.classes) <= 2 else 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    callbacks = get_callbacks(config, 'finetune')
    return model.fit(train_gen, validation_data=val_gen, epochs=config.epochs - config.freeze_base_epochs,
                     class_weight=class_weight, callbacks=callbacks)

def plot_training_history(h1, h2, output_dir):
    output_dir = Path(output_dir)
    plt.figure()
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history.get('val_accuracy', []) + h2.history.get('val_accuracy', [])
    plt.plot(acc, label='train_acc')
    if val_acc:
        plt.plot(val_acc, label='val_acc')
    plt.title('Training Accuracy')
    plt.legend()
    plt.savefig(output_dir / "accuracy_curve.png")

def train(config: ResNet50Config):
    paths, labels = load_dataset(config.data_dir, config.classes)
    if len(paths) == 0:
        logging.warning("No data found for training classifier.")
        return
        
    X_train, X_val, y_train, y_val = train_test_split(paths, labels, test_size=config.validation_split)
    
    classes_unique = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes_unique, y=y_train)
    class_weight = {i: w for i, w in zip(classes_unique, cw)}
    
    train_gen = ClassificationDataGenerator(X_train, y_train, config.batch_size, config.input_size, config.augmentation)
    val_gen = ClassificationDataGenerator(X_val, y_val, config.batch_size, config.input_size, False)
    
    model = build_resnet50(config.input_size, len(config.classes), freeze_base=True)
    h1 = phase1_train(model, train_gen, val_gen, config, class_weight)
    h2 = phase2_finetune(model, train_gen, val_gen, config, class_weight)
    
    model.save(config.checkpoint_path)
    plot_training_history(h1, h2, Path(config.checkpoint_path).parent)
    logging.info("Training complete")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=resnet_config.epochs)
    parser.add_argument("--batch_size", type=int, default=resnet_config.batch_size)
    return parser.parse_args()

def main():
    args = parse_args()
    config = resnet_config
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    train(config)

if __name__ == "__main__":
    main()
