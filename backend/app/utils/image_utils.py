import base64
import uuid
import cv2
import numpy as np
import pydicom
from io import BytesIO
from PIL import Image
from fastapi import HTTPException
from typing import Tuple
from pathlib import Path

def validate_file(filename: str, content: bytes, max_size: int) -> None:
    if not filename or not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty or has no filename")
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="The uploaded file exceeds the maximum allowed size")
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ["png", "jpg", "jpeg", "dcm"]:
        raise HTTPException(status_code=400, detail="This file format is not supported")

def load_image_from_bytes(content: bytes, filename: str) -> np.ndarray:
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext == "dcm":
        try:
            dicom = pydicom.dcmread(BytesIO(content))
            img = dicom.pixel_array
            if "RescaleSlope" in dicom and "RescaleIntercept" in dicom:
                img = img * float(dicom.RescaleSlope) + float(dicom.RescaleIntercept)
            img = img - np.min(img)
            max_value = np.max(img)
            if max_value > 0:
                img = img / max_value
            else:
                img = np.zeros_like(img)
            return (img * 255).astype(np.uint8)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read DICOM file: {str(e)}")
    else:
        # Try OpenCV first
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Fallback to PIL
        try:
            pil_img = Image.open(BytesIO(content)).convert("RGB")
            return np.array(pil_img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read image file: {str(e)}")

def numpy_to_base64(img: np.ndarray, format: str = "PNG") -> str:
    if img is None:
        return ""
    # Convert float image to uint8 if necessary
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
            
    success, encoded_image = cv2.imencode(f".{format.lower()}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img)
    if not success:
        raise ValueError("Failed to encode image")
    b64_string = base64.b64encode(encoded_image).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64_string}"

def generate_unique_id() -> str:
    return str(uuid.uuid4())

def save_image(img: np.ndarray, directory: str, filename: str) -> str:
    path = Path(directory) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(img.shape) == 3:
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(path), img)
    return str(path)

def create_overlay(original: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    original = ensure_rgb(original)
    overlay = original.copy()
    colored_mask = np.zeros_like(original)
    for c in range(3):
        colored_mask[:, :, c] = mask * color[c]
    
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    
    # Only apply overlay where mask > 0
    mask_indices = mask > 0
    result = original.copy()
    result[mask_indices] = overlay[mask_indices]
    return result

def resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def ensure_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img
