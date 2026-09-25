"""
Model Evaluation Script

Evaluates trained U-Net and ResNet50 models on test data and saves real metrics.

Usage:
    python training/evaluate.py --model resnet50 --test_dir datasets/classification/test
    python training/evaluate.py --model unet --test_dir datasets/processed/test
    python training/evaluate.py --model all

Outputs:
    training/outputs/evaluation/
        classification_metrics.json
        segmentation_metrics.json
        confusion_matrix.png
        roc_curve.png

NOTE: Metrics are only computed when model weights AND test data are present.
      If either is missing, the script exits with a clear error — it never
      fabricates performance numbers.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import tensorflow as tf
from training.config import resnet_config, unet_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_classification_test_data(
    test_data_dir: str,
    classes: list[str],
    input_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load images from class subdirectories for classification evaluation."""
    data_dir = Path(test_data_dir)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for class_idx, class_name in enumerate(classes):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}\n"
                f"Expected structure: {data_dir}/<class_name>/*.png"
            )
        img_paths = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
        if not img_paths:
            logger.warning(f"No images found in {class_dir}")
            continue
        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None:
                logger.warning(f"Could not read image: {p}")
                continue
            img = cv2.resize(img, input_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img.astype(np.float32) / 255.0)
            labels.append(class_idx)

    if not images:
        raise ValueError(f"No valid images found under {test_data_dir}")

    return np.array(images), np.array(labels)


def _load_segmentation_test_data(
    test_data_dir: str,
    input_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load paired image/mask PNGs for segmentation evaluation."""
    data_dir = Path(test_data_dir)
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"

    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Expected 'images/' and 'masks/' subdirectories under {test_data_dir}"
        )

    img_paths = sorted(img_dir.glob("*.png"))
    if not img_paths:
        raise ValueError(f"No PNG images found in {img_dir}")

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for img_path in img_paths:
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            logger.warning(f"No corresponding mask for {img_path.name}, skipping.")
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        img = cv2.resize(img, input_size)
        mask = cv2.resize(mask, input_size)
        images.append((img.astype(np.float32) / 255.0)[..., np.newaxis])
        masks.append(((mask > 127).astype(np.float32))[..., np.newaxis])

    if not images:
        raise ValueError(f"No valid image/mask pairs found under {test_data_dir}")

    return np.array(images), np.array(masks)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info(f"ROC curve saved to {output_path}")


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

def evaluate_classifier(
    model_path: str,
    test_data_dir: str,
    config: Any,
) -> dict[str, Any]:
    """
    Load a trained ResNet50 classifier and evaluate it on real test data.

    Returns a dict with accuracy, precision, recall, f1, roc_auc.
    Raises if model weights or test data are missing.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"ResNet50 weights not found at {model_path}.\n"
            "Train the model first using: python training/train_resnet50.py"
        )

    logger.info(f"Loading classifier from {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    logger.info(f"Loading test data from {test_data_dir}")
    input_size = (config.input_size[1], config.input_size[0])  # (W, H) for cv2
    images, y_true = _load_classification_test_data(test_data_dir, config.classes, input_size)

    logger.info(f"Running inference on {len(images)} test images ...")
    probs = model.predict(images, batch_size=16, verbose=0)  # shape (N, 2)
    y_pred = np.argmax(probs, axis=1)
    y_prob_positive = probs[:, 1]  # probability of positive class

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="binary", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob_positive))
        fpr, tpr, _ = roc_curve(y_true, y_prob_positive)
    except ValueError as e:
        logger.warning(f"Could not compute ROC-AUC: {e}")
        auc = None
        fpr, tpr = None, None

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=config.classes)
    logger.info(f"\n{report}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "_fpr": fpr.tolist() if fpr is not None else None,
        "_tpr": tpr.tolist() if tpr is not None else None,
        "num_test_samples": num_test_samples,
        "classes": config.classes,
    }


def evaluate_segmenter(
    model_path: str,
    test_data_dir: str,
    config: Any,
) -> dict[str, Any]:
    """
    Load a trained U-Net segmenter and evaluate it on real test data.

    Returns a dict with dice coefficient and IoU.
    Raises if model weights or test data are missing.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"U-Net weights not found at {model_path}.\n"
            "Train the model first using: python training/train_unet.py"
        )

    logger.info(f"Loading U-Net from {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    input_size_hw = (config.input_size[0], config.input_size[1])  # (H, W)
    input_size_wh = (config.input_size[1], config.input_size[0])  # for cv2

    logger.info(f"Loading segmentation test data from {test_data_dir}")

    data_dir = Path(test_data_dir)
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"

    img_paths = sorted(img_dir.glob("*.png"))
    if not img_paths:
        raise ValueError(f"No test images found in {img_dir}")

    # Process one image at a time to avoid loading the entire test set into RAM.
    batch_size = 1
    smooth = 1e-6
    dice_scores = []
    iou_scores = []
    num_test_samples = 0

    logger.info(
        f"Running segmentation on {len(img_paths)} images "
        f"one at a time to limit memory usage..."
    )

    for img_path in img_paths:
        mask_path = mask_dir / img_path.name

        if not mask_path.exists():
            logger.warning(f"Missing mask for {img_path.name}, skipping")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            logger.warning(f"Could not read {img_path.name}, skipping")
            continue

        img = cv2.resize(img, input_size_wh)
        mask = cv2.resize(mask, input_size_wh)

        image = (img.astype(np.float32) / 255.0)[np.newaxis, ..., np.newaxis]
        mask = ((mask > 127).astype(np.float32))[np.newaxis, ..., np.newaxis]

        pred = model.predict(image, batch_size=batch_size, verbose=0)
        pred = (pred > 0.5).astype(np.float32)

        pred_flat = pred[0].reshape(-1)
        mask_flat = mask[0].reshape(-1)

        intersection = np.sum(pred_flat * mask_flat)
        dice = (2.0 * intersection + smooth) / (
            np.sum(pred_flat) + np.sum(mask_flat) + smooth
        )

        union = np.sum(pred_flat) + np.sum(mask_flat) - intersection
        iou = (intersection + smooth) / (union + smooth)

        dice_scores.append(float(dice))
        iou_scores.append(float(iou))
        num_test_samples += 1

        if num_test_samples % 250 == 0:
            logger.info(
                f"Processed {num_test_samples}/{len(img_paths)} test images"
            )

    if num_test_samples == 0:
        raise ValueError("No valid test image/mask pairs were processed.")

    mean_dice = float(np.mean(dice_scores))
    mean_iou = float(np.mean(iou_scores))

    logger.info(f"Test samples evaluated: {num_test_samples}")
    logger.info(f"Mean Dice: {mean_dice:.6f}")
    logger.info(f"Mean IoU: {mean_iou:.6f}")

    return {
        "mean_dice_coefficient": mean_dice,
        "mean_iou": mean_iou,
        "std_dice": float(np.std(dice_scores)),
        "std_iou": float(np.std(iou_scores)),
        "num_test_samples": num_test_samples,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """Persist metrics to JSON, excluding internal plotting arrays."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in metrics.items() if not k.startswith("_")}
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=4)
    logger.info(f"Metrics saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained U-Net and/or ResNet50 models on real test data."
    )
    parser.add_argument(
        "--model",
        choices=["resnet50", "unet", "all"],
        default="all",
        help="Which model to evaluate",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Override test data directory (auto-detected if not set)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(__file__).parent / "outputs" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).parent.parent
    default_cls_test = str(root / "datasets" / "classification" / "test")
    default_seg_test = str(root / "datasets" / "processed" / "test")

    errors: list[str] = []

    if args.model in ("resnet50", "all"):
        test_dir = args.test_dir or default_cls_test
        try:
            metrics = evaluate_classifier(
                resnet_config.checkpoint_path, test_dir, resnet_config
            )
            save_metrics(metrics, out_dir / "classification_metrics.json")
            # Plots
            cm = np.array(metrics["confusion_matrix"])
            plot_confusion_matrix(cm, resnet_config.classes, out_dir / "confusion_matrix.png")
            if metrics.get("_fpr") is not None:
                plot_roc_curve(
                    np.array(metrics["_fpr"]),
                    np.array(metrics["_tpr"]),
                    metrics["roc_auc"],
                    out_dir / "roc_curve.png",
                )
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Classification evaluation failed: {e}")
            errors.append(str(e))

    if args.model in ("unet", "all"):
        test_dir = args.test_dir or default_seg_test
        try:
            metrics = evaluate_segmenter(
                unet_config.checkpoint_path, test_dir, unet_config
            )
            save_metrics(metrics, out_dir / "segmentation_metrics.json")
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Segmentation evaluation failed: {e}")
            errors.append(str(e))

    if errors:
        logger.warning(
            "\n\nEvaluation could not complete for one or more models:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n\nTrain the models first, then re-run evaluation."
        )
        sys.exit(1)
    else:
        logger.info(f"\n✅ Evaluation complete. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
