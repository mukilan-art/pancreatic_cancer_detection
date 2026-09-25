"""
TCIA Pancreas-CT Preprocessing & DICOM + NIfTI Annotation Conversion Script

Converts raw TCIA Pancreas-CT DICOM series and 3D NIfTI segmentation masks (labelXXXX.nii.gz)
into paired 512x512 PNG CT images (grayscale with CLAHE) and binary PNG masks (0=background, 255=pancreas).

Features:
- Discovers PANCREAS_XXXX patient DICOM series automatically.
- Sorts DICOM slices spatially using ImagePositionPatient and ImageOrientationPatient.
- Aligns 3D NIfTI label volumes (labelXXXX.nii.gz) with sorted DICOM slices.
- Applies standard abdominal CT windowing (-100 to 240 HU) and CLAHE enhancement.
- Resizes both CT slices and binary masks to 512x512.
- Performs patient-level train / validation / test partitioning (preventing cross-slice data leakage).
- Exports a JSON manifest of the split for auditability.
- Supports dry-run and patient limiting for verification.

Usage:
    # Dry run to verify matching and series detection:
    python training/preprocess_dataset.py --dry_run

    # Full preprocessing:
    python training/preprocess_dataset.py --dicom_dir pancreas_ct --labels_dir datasets/pancreas-ct/labels --output_dir datasets/processed
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    from backend.app.preprocessing.clahe_processor import CLAHEProcessor
except ImportError:
    class CLAHEProcessor:
        def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
            self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))

        def apply_clahe(self, image: np.ndarray) -> np.ndarray:
            return self.clahe.apply(image)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pancreas_preprocessor")


# ---------------------------------------------------------------------------
# DICOM Processing Helpers
# ---------------------------------------------------------------------------

def compute_slice_z_position(ds: "pydicom.dataset.Dataset") -> float:
    """
    Computes the position along the scan normal vector for accurate 3D spatial sorting.
    Falls back to ImagePositionPatient[2], SliceLocation, or InstanceNumber if needed.
    """
    try:
        iop = [float(x) for x in ds.ImageOrientationPatient]
        ipp = [float(x) for x in ds.ImagePositionPatient]
        # Cross product of row and column direction vectors gives normal to slice
        row_vec = np.array(iop[:3], dtype=np.float64)
        col_vec = np.array(iop[3:], dtype=np.float64)
        normal = np.cross(row_vec, col_vec)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
            return float(np.dot(np.array(ipp, dtype=np.float64), normal))
        return float(ipp[2])
    except Exception:
        if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
            return float(ds.ImagePositionPatient[2])
        if hasattr(ds, "SliceLocation"):
            return float(ds.SliceLocation)
        if hasattr(ds, "InstanceNumber"):
            return float(ds.InstanceNumber)
        return 0.0


def read_and_sort_dicom_series(patient_dir: Path) -> List[Tuple[Path, float]]:
    """
    Finds all DICOM files under patient_dir and sorts them spatially.
    Returns list of (file_path, z_position).
    """
    if pydicom is None:
        raise ImportError("pydicom is required. Install via: pip install pydicom")

    dcm_files = list(patient_dir.rglob("*.dcm"))
    if not dcm_files:
        return []

    slice_entries = []
    for fpath in dcm_files:
        try:
            ds = pydicom.dcmread(str(fpath), stop_before_pixels=True)
            z_pos = compute_slice_z_position(ds)
            slice_entries.append((fpath, z_pos))
        except Exception as e:
            logger.warning(f"Failed to read DICOM header for {fpath.name}: {e}")

    # Sort ascending by z position
    slice_entries.sort(key=lambda item: item[1])
    return slice_entries


def dicom_to_hu_image(
    dcm_path: Path,
    window_min: float = -100.0,
    window_max: float = 240.0,
    target_size: int = 512,
    clahe_proc: Optional[CLAHEProcessor] = None,
) -> Optional[np.ndarray]:
    """
    Reads a DICOM slice, converts to Hounsfield Units (HU), applies abdominal windowing,
    scales to [0, 255] uint8, applies CLAHE contrast enhancement, and resizes.
    """
    try:
        ds = pydicom.dcmread(str(dcm_path))
        pixel_array = ds.pixel_array.astype(np.float32)

        # Apply Rescale Slope & Intercept for Hounsfield Units (HU)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = pixel_array * slope + intercept

        # Apply Abdominal Soft-Tissue Windowing
        hu_clipped = np.clip(hu, window_min, window_max)
        hu_normalized = (hu_clipped - window_min) / (window_max - window_min + 1e-8) * 255.0
        img_uint8 = hu_normalized.astype(np.uint8)

        # Apply CLAHE Contrast Enhancement
        if clahe_proc is not None:
            img_uint8 = clahe_proc.apply_clahe(img_uint8)

        # Resize to target 512x512
        if img_uint8.shape[:2] != (target_size, target_size):
            img_uint8 = cv2.resize(img_uint8, (target_size, target_size), interpolation=cv2.INTER_AREA)

        return img_uint8
    except Exception as e:
        logger.error(f"Error processing DICOM slice {dcm_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# NIfTI Label Processing Helpers
# ---------------------------------------------------------------------------

def extract_patient_id(text: str) -> Optional[str]:
    """
    Extracts 4-digit patient identifier (e.g. '0001' from 'PANCREAS_0001' or 'label0001.nii.gz').
    """
    match = re.search(r"(\d{4})", text)
    return match.group(1) if match else None


def find_matching_label_file(patient_id: str, labels_dir: Path) -> Optional[Path]:
    """
    Searches labels_dir for the NIfTI label file corresponding to a patient ID (e.g. '0001').
    """
    candidates = [
        f"label{patient_id}.nii.gz",
        f"label_{patient_id}.nii.gz",
        f"label{patient_id}.nii",
        f"label_{patient_id}.nii",
        f"PANCREAS_{patient_id}.nii.gz",
        f"PANCREAS_{patient_id}.nii",
        f"pancreas_{patient_id}.nii.gz",
        f"pancreas_{patient_id}.nii",
    ]

    for cand in candidates:
        target = labels_dir / cand
        if target.exists():
            return target

    # Fallback: search case-insensitively with regex in labels_dir
    if labels_dir.exists():
        for p in labels_dir.glob("*.*"):
            if patient_id in p.name and (".nii" in p.name.lower()):
                return p

    return None


def load_nifti_volume(nii_path: Path) -> Optional[np.ndarray]:
    """
    Loads 3D NIfTI label volume into a numpy array.
    """
    if nib is None:
        raise ImportError("nibabel is required. Install via: pip install nibabel")

    try:
        nii = nib.load(str(nii_path))
        data = np.asanyarray(nii.dataobj)
        return data
    except Exception as e:
        logger.error(f"Error loading NIfTI volume {nii_path}: {e}")
        return None


def extract_nifti_slice(
    label_vol: np.ndarray,
    slice_idx: int,
    total_slices: int,
    target_size: int = 512,
) -> Optional[np.ndarray]:
    """
    Extracts corresponding 2D binary mask slice from 3D NIfTI label volume,
    orients it to match the DICOM axial plane, and resizes to 512x512.
    """
    try:
        # Determine which dimension corresponds to the slice axis
        vol_shape = label_vol.shape
        if len(vol_shape) != 3:
            logger.error(f"Expected 3D NIfTI volume, got shape {vol_shape}")
            return None

        # Identify slice dimension matching total_slices (usually axis 2 in TCIA)
        slice_axis = 2
        if vol_shape[2] == total_slices:
            slice_axis = 2
        elif vol_shape[0] == total_slices:
            slice_axis = 0
        elif vol_shape[1] == total_slices:
            slice_axis = 1

        if slice_axis == 2:
            mask_slice = label_vol[:, :, slice_idx]
        elif slice_axis == 0:
            mask_slice = label_vol[slice_idx, :, :]
        else:
            mask_slice = label_vol[:, slice_idx, :]

        # Standard TCIA Pancreas-CT NIfTI to DICOM array orientation:
        # NIfTI is (X, Y) where DICOM image is (rows=Y, cols=X). Transpose/rotate to align.
        # In TCIA annotations, rotating 90 deg clockwise or transposing matches axial DICOM coordinates.
        mask_slice = mask_slice.T

        # Convert to strict binary uint8 (0=background, 255=pancreas)
        binary_mask = (mask_slice > 0).astype(np.uint8) * 255

        # Resize with NEAREST interpolation to prevent blurred / non-binary edges
        if binary_mask.shape[:2] != (target_size, target_size):
            binary_mask = cv2.resize(
                binary_mask,
                (target_size, target_size),
                interpolation=cv2.INTER_NEAREST,
            )

        return binary_mask
    except Exception as e:
        logger.error(f"Error extracting slice {slice_idx} from NIfTI volume: {e}")
        return None


# ---------------------------------------------------------------------------
# Patient-Level Dataset Splitting & Export
# ---------------------------------------------------------------------------

def split_patients(
    patient_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Partitions patient IDs into train, validation, and test splits at the PATIENT level.
    """
    sorted_ids = sorted(patient_ids)
    rng = np.random.RandomState(seed)
    shuffled_ids = list(sorted_ids)
    rng.shuffle(shuffled_ids)

    n_total = len(shuffled_ids)
    n_train = int(np.floor(train_ratio * n_total))
    n_val = int(np.floor(val_ratio * n_total))

    train_ids = sorted(shuffled_ids[:n_train])
    val_ids = sorted(shuffled_ids[n_train : n_train + n_val])
    test_ids = sorted(shuffled_ids[n_train + n_val :])

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


# ---------------------------------------------------------------------------
# Main Preprocessing Pipeline
# ---------------------------------------------------------------------------

def preprocess_pancreas_dataset(
    dicom_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    target_size: int = 512,
    window_min: float = -100.0,
    window_max: float = 240.0,
    seed: int = 42,
    dry_run: bool = False,
    limit_patients: Optional[int] = None,
) -> Dict[str, int]:
    """
    Executes end-to-end DICOM + NIfTI conversion and dataset splitting.
    """
    logger.info("=" * 70)
    logger.info("TCIA Pancreas-CT Dataset Preprocessing Pipeline")
    logger.info("=" * 70)
    logger.info(f"DICOM Directory   : {dicom_dir}")
    logger.info(f"Labels Directory  : {labels_dir}")
    logger.info(f"Output Directory  : {output_dir}")
    logger.info(f"Patient Splits    : Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
    logger.info(f"Target Resolution : {target_size}x{target_size} PNG")
    logger.info(f"HU Window Range   : [{window_min}, {window_max}] HU")
    logger.info(f"Dry Run Mode      : {dry_run}")
    logger.info("=" * 70)

    # 1. Discover all patient folders in DICOM directory
    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    patient_folders = [
        p for p in sorted(dicom_dir.iterdir())
        if p.is_dir() and "PANCREAS_" in p.name.upper()
    ]

    if not patient_folders:
        # Check subfolders or flat structure
        patient_folders = [p for p in sorted(dicom_dir.iterdir()) if p.is_dir()]

    logger.info(f"Found {len(patient_folders)} candidate patient folders in {dicom_dir.name}")

    # 2. Match patients with corresponding NIfTI label files
    matched_patients: List[Dict] = []
    missing_labels: List[str] = []

    for p_dir in patient_folders:
        pat_id = extract_patient_id(p_dir.name)
        if not pat_id:
            continue

        label_path = find_matching_label_file(pat_id, labels_dir)
        if label_path is not None and label_path.exists():
            matched_patients.append({
                "patient_name": f"PANCREAS_{pat_id}",
                "patient_id": pat_id,
                "dicom_dir": p_dir,
                "label_path": label_path,
            })
        else:
            missing_labels.append(f"PANCREAS_{pat_id}")

    logger.info(f"Matched {len(matched_patients)} patients with NIfTI label files.")
    if missing_labels:
        logger.warning(
            f"{len(missing_labels)} patients have missing labels in {labels_dir}: "
            f"{', '.join(missing_labels[:10])}{'...' if len(missing_labels) > 10 else ''}"
        )

    if not matched_patients:
        logger.error(
            f"No matched DICOM series and NIfTI label files found.\n"
            f"Please ensure label files (e.g. label0001.nii.gz) are placed in: {labels_dir}"
        )
        return {"total_patients": 0, "processed_slices": 0}

    if limit_patients is not None and limit_patients > 0:
        matched_patients = matched_patients[:limit_patients]
        logger.info(f"Limiting execution to first {limit_patients} patients.")

    # 3. Patient-Level Splitting
    patient_names = [p["patient_name"] for p in matched_patients]
    split_map = split_patients(
        patient_names,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    logger.info("-" * 70)
    logger.info(f"Patient Allocation:")
    logger.info(f"  Train : {len(split_map['train'])} patients -> {split_map['train']}")
    logger.info(f"  Val   : {len(split_map['val'])} patients -> {split_map['val']}")
    logger.info(f"  Test  : {len(split_map['test'])} patients -> {split_map['test']}")
    logger.info("-" * 70)

    # In dry-run mode, inspect series details and exit
    if dry_run:
        logger.info("[DRY RUN] Inspecting first 3 matched volumes for alignment...")
        for p_info in matched_patients[:3]:
            dicom_slices = read_and_sort_dicom_series(p_info["dicom_dir"])
            label_vol = load_nifti_volume(p_info["label_path"])
            vol_shape = label_vol.shape if label_vol is not None else "None"
            logger.info(
                f"  Patient {p_info['patient_name']}: "
                f"{len(dicom_slices)} DICOM slices, NIfTI shape: {vol_shape}"
            )
        logger.info("[DRY RUN] Complete. No files written.")
        return {
            "matched_patients": len(matched_patients),
            "train_patients": len(split_map["train"]),
            "val_patients": len(split_map["val"]),
            "test_patients": len(split_map["test"]),
        }

    # 4. Create output directory tree
    for split_name in ["train", "val", "test"]:
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "masks").mkdir(parents=True, exist_ok=True)

    # Save split manifest for reproducibility and auditability
    manifest_path = output_dir / "dataset_split.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "splits": split_map,
                "train_count": len(split_map["train"]),
                "val_count": len(split_map["val"]),
                "test_count": len(split_map["test"]),
                "random_seed": seed,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved dataset split manifest to {manifest_path}")

    # 5. Process matched volumes
    clahe_proc = CLAHEProcessor(clip_limit=2.0, tile_grid_size=8)
    stats = {"train": 0, "val": 0, "test": 0, "total_slices": 0, "positive_pancreas_slices": 0}

    # Build reverse lookup for patient -> split
    patient_to_split = {}
    for s_name, p_list in split_map.items():
        for p_name in p_list:
            patient_to_split[p_name] = s_name

    for idx, p_info in enumerate(matched_patients, 1):
        p_name = p_info["patient_name"]
        split_name = patient_to_split.get(p_name, "train")
        split_img_dir = output_dir / split_name / "images"
        split_mask_dir = output_dir / split_name / "masks"

        logger.info(f"[{idx}/{len(matched_patients)}] Processing {p_name} -> {split_name}/...")

        # Sort DICOM slices along scan normal vector
        dicom_slices = read_and_sort_dicom_series(p_info["dicom_dir"])
        if not dicom_slices:
            logger.warning(f"No DICOM slices found for {p_name}")
            continue

        # Load 3D NIfTI label volume
        label_vol = load_nifti_volume(p_info["label_path"])
        if label_vol is None:
            logger.warning(f"Failed to load NIfTI label for {p_name}")
            continue

        num_slices = len(dicom_slices)
        for s_idx, (dcm_path, z_pos) in enumerate(dicom_slices):
            # Process CT slice
            ct_img = dicom_to_hu_image(
                dcm_path,
                window_min=window_min,
                window_max=window_max,
                target_size=target_size,
                clahe_proc=clahe_proc,
            )
            if ct_img is None:
                continue

            # Extract aligned label slice
            mask_img = extract_nifti_slice(
                label_vol=label_vol,
                slice_idx=s_idx,
                total_slices=num_slices,
                target_size=target_size,
            )
            if mask_img is None:
                continue

            # File naming: Identical for both CT image and mask
            filename = f"{p_name}_slice_{s_idx:04d}.png"
            img_out_path = split_img_dir / filename
            mask_out_path = split_mask_dir / filename

            # Write PNGs
            cv2.imwrite(str(img_out_path), ct_img)
            cv2.imwrite(str(mask_out_path), mask_img)

            stats[split_name] += 1
            stats["total_slices"] += 1
            if np.count_nonzero(mask_img) > 0:
                stats["positive_pancreas_slices"] += 1

    logger.info("=" * 70)
    logger.info("Preprocessing & Conversion Complete!")
    logger.info(f"Total Slices Processed       : {stats['total_slices']}")
    logger.info(f"Slices with Pancreas Visible : {stats['positive_pancreas_slices']}")
    logger.info(f"  - Train Set : {stats['train']} slice pairs")
    logger.info(f"  - Val Set   : {stats['val']} slice pairs")
    logger.info(f"  - Test Set  : {stats['test']} slice pairs")
    logger.info(f"Output saved to: {output_dir}")
    logger.info("=" * 70)

    return stats


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TCIA Pancreas-CT DICOM series and 3D NIfTI labels into 512x512 PNG pairs for U-Net training."
    )
    parser.add_argument(
        "--dicom_dir",
        type=str,
        default="pancreas_ct",
        help="Path to folder containing PANCREAS_XXXX DICOM directories (default: pancreas_ct)",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default="datasets/pancreas-ct/labels",
        help="Path to folder containing labelXXXX.nii.gz files (default: datasets/pancreas-ct/labels)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/processed",
        help="Destination directory for processed PNG dataset (default: datasets/processed)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Fraction of patients allocated to train split (default: 0.70)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction of patients allocated to validation split (default: 0.15)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Fraction of patients allocated to test split (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patient-level split (default: 42)",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="Target square image resolution (default: 512)",
    )
    parser.add_argument(
        "--window_min",
        type=float,
        default=-100.0,
        help="Minimum HU for abdominal windowing (default: -100.0)",
    )
    parser.add_argument(
        "--window_max",
        type=float,
        default=240.0,
        help="Maximum HU for abdominal windowing (default: 240.0)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform dry-run to check matching without writing files",
    )
    parser.add_argument(
        "--limit_patients",
        type=int,
        default=None,
        help="Limit preprocessing to first N patients (useful for testing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dicom_path = Path(args.dicom_dir)
    if not dicom_path.is_absolute():
        dicom_path = ROOT_DIR / dicom_path

    labels_path = Path(args.labels_dir)
    if not labels_path.is_absolute():
        labels_path = ROOT_DIR / labels_path

    output_path = Path(args.output_dir)
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path

    preprocess_pancreas_dataset(
        dicom_dir=dicom_path,
        labels_dir=labels_path,
        output_dir=output_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        target_size=args.target_size,
        window_min=args.window_min,
        window_max=args.window_max,
        seed=args.seed,
        dry_run=args.dry_run,
        limit_patients=args.limit_patients,
    )


if __name__ == "__main__":
    main()
