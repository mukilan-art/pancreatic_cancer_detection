"""
U-Net Training Script for Pancreas Segmentation

Usage:
    python training/train_unet.py
    python training/train_unet.py --epochs 50 --batch_size 2

Dataset structure expected:
    datasets/processed/
        train/
            images/  (*.png, 512x512 grayscale)
            masks/   (*.png, 512x512 binary)
        val/
            images/
            masks/
"""
import argparse
import logging
import sys
import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')
from backend.app.segmentation.unet_model import build_unet

from training.config import UNetConfig, unet_config

logging.basicConfig(level=logging.INFO)

class PancreasDataGenerator(keras.utils.Sequence):
    def __init__(self, img_paths, mask_paths, batch_size, img_size, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_img_paths = self.img_paths[i : i + self.batch_size]
        batch_mask_paths = self.mask_paths[i : i + self.batch_size]

        x = np.zeros((self.batch_size, *self.img_size), dtype="float32")
        y = np.zeros((self.batch_size, *self.img_size), dtype="float32")

        for j, (img_p, mask_p) in enumerate(zip(batch_img_paths, batch_mask_paths)):
            img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, self.img_size[:2])
            mask = cv2.resize(mask, self.img_size[:2], interpolation=cv2.INTER_NEAREST)

            if self.augment:
                if np.random.rand() > 0.5:
                    img = np.fliplr(img)
                    mask = np.fliplr(mask)

            x[j] = np.expand_dims(img / 255.0, axis=-1)
            y[j] = np.expand_dims((mask > 127).astype("float32"), axis=-1)

        return x, y


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.keras.backend.sum(
        tf.keras.backend.abs(y_true * y_pred), axis=[1, 2, 3]
    )
    union = (
        tf.keras.backend.sum(y_true, [1, 2, 3])
        + tf.keras.backend.sum(y_pred, [1, 2, 3])
        - intersection
    )
    return tf.keras.backend.mean(
        (intersection + smooth) / (union + smooth), axis=0
    )

def get_callbacks(config: UNetConfig):
    checkpoint_dir = Path(config.checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(config.checkpoint_path, save_best_only=True),
        keras.callbacks.EarlyStopping(patience=config.early_stopping_patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=config.reduce_lr_patience, factor=0.5, min_lr=1e-6),
        keras.callbacks.CSVLogger(str(checkpoint_dir / "training.log"))
    ]

def plot_training_history(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()
    
def train(config: UNetConfig, smoke_test: bool = False):
    logging.info("Starting UNet training")
    data_dir = Path(config.data_dir)
    
    train_img_dir = data_dir / "train" / "images"
    train_mask_dir = data_dir / "train" / "masks"

    train_imgs = []
    train_masks = []

    for img_path in sorted(train_img_dir.glob("*.png")):
        mask_path = train_mask_dir / img_path.name

        if mask_path.exists():
            train_imgs.append(img_path)
            train_masks.append(mask_path)

    print(f"Training image-mask pairs: {len(train_imgs)}")

    val_img_dir = data_dir / "val" / "images"
    val_mask_dir = data_dir / "val" / "masks"

    val_imgs = []
    val_masks = []

    for img_path in sorted(val_img_dir.glob("*.png")):
        mask_path = val_mask_dir / img_path.name

        if mask_path.exists():
            val_imgs.append(img_path)
            val_masks.append(mask_path)

    print(f"Validation image-mask pairs: {len(val_imgs)}")

    if smoke_test:
        logging.info("--- RUNNING IN SMOKE TEST MODE ---")
        def select_smoke_subset(img_list, mask_list, target_non_empty=4, target_empty=4):
            non_empty_pairs = []
            empty_pairs = []
            for img_p, mask_p in zip(img_list, mask_list):
                m = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
                if m is not None and np.count_nonzero(m) > 0:
                    if len(non_empty_pairs) < target_non_empty:
                        non_empty_pairs.append((img_p, mask_p))
                else:
                    if len(empty_pairs) < target_empty:
                        empty_pairs.append((img_p, mask_p))
                if len(non_empty_pairs) >= target_non_empty and len(empty_pairs) >= target_empty:
                    break
            # Interleave positive and negative pairs for batch balance
            selected = [p for pair in zip(non_empty_pairs, empty_pairs) for p in pair]
            return [p[0] for p in selected], [p[1] for p in selected]

        target_count = max(4, config.batch_size)
        train_imgs, train_masks = select_smoke_subset(train_imgs, train_masks, target_non_empty=target_count, target_empty=target_count)
        val_imgs, val_masks = select_smoke_subset(val_imgs, val_masks, target_non_empty=target_count, target_empty=target_count)
        config.epochs = 1
        print("Selected Smoke Test Training Files:")
        for ip, mp in zip(train_imgs, train_masks):
            nz = int(np.count_nonzero(cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)))
            print(f"  Train: {ip.name} | Non-zero pixels: {nz} ({'Non-empty' if nz>0 else 'Empty'})")
        print("Selected Smoke Test Validation Files:")
        for ip, mp in zip(val_imgs, val_masks):
            nz = int(np.count_nonzero(cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)))
            print(f"  Val:   {ip.name} | Non-zero pixels: {nz} ({'Non-empty' if nz>0 else 'Empty'})")

    train_gen = PancreasDataGenerator(train_imgs, train_masks, config.batch_size, config.input_size, config.augmentation)
    val_gen = PancreasDataGenerator(
        val_imgs,
        val_masks,
        config.batch_size,
        config.input_size,
        augment=False
    )
    
    model = build_unet(config.input_size)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
                  loss=bce_dice_loss, metrics=[dice_coefficient, iou_metric])
                  
    callbacks = get_callbacks(config)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.epochs,
        callbacks=callbacks
    )
    
    plot_training_history(history, str(Path(config.checkpoint_path).parent))
    logging.info("Training complete")
    if 'loss' in history.history:
        print(f"Final Metrics - loss: {history.history['loss'][-1]:.6f}, "
              f"dice_coefficient: {history.history['dice_coefficient'][-1]:.6f}, "
              f"iou_metric: {history.history['iou_metric'][-1]:.6f}")
    if 'val_loss' in history.history:
        print(f"Final Metrics - val_loss: {history.history['val_loss'][-1]:.6f}, "
              f"val_dice_coefficient: {history.history['val_dice_coefficient'][-1]:.6f}, "
              f"val_iou_metric: {history.history['val_iou_metric'][-1]:.6f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net for Pancreas Segmentation")
    parser.add_argument("--epochs", type=int, default=unet_config.epochs)
    parser.add_argument("--batch_size", type=int, default=unet_config.batch_size)
    parser.add_argument("--learning_rate", type=float, default=unet_config.learning_rate)
    parser.add_argument("--data_dir", type=str, default=unet_config.data_dir)
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick 1-epoch smoke test on a small subset of batches")
    return parser.parse_args()

def main():
    args = parse_args()
    config = unet_config
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.data_dir = args.data_dir
    train(config, smoke_test=args.smoke_test)

if __name__ == "__main__":
    main()
