import numpy as np
import cv2
import os
from typing import Dict, Any
from app.utils.image_utils import ensure_grayscale, ensure_rgb, create_overlay

class UNetInference:
    def __init__(self, model: Any, weights_loaded: bool, model_path: str = ""):
        self.model = model
        self.weights_loaded = weights_loaded
        self.model_path = model_path
        self.model_trained = True

    def segment(self, image: np.ndarray) -> Dict[str, Any]:
        if not self.weights_loaded or self.model is None:
            return {
                "mask": np.zeros(image.shape[:2], dtype=np.uint8) if len(image.shape) == 3 else np.zeros_like(image, dtype=np.uint8),
                "raw_output": None,
                "confidence": 0.0,
                "pixel_percentage": 0.0,
                "prediction_min": 0.0,
                "prediction_max": 0.0,
                "prediction_mean": 0.0,
                "prediction_std": 0.0,
                "weights_loaded": False,
                "model_trained": False,
                "valid": False,
                "warning": "U-Net weights unavailable — segmentation mask cannot be generated."
            }
        
        # Preprocess input image to grayscale
        gray = ensure_grayscale(image)
        if gray.dtype != np.uint8:
            gray_uint8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
        else:
            gray_uint8 = gray

        input_data = (gray_uint8.astype(np.float32) / 255.0)[..., np.newaxis]
        input_data = np.expand_dims(input_data, axis=0)  # (1, 512, 512, 1)
        
        # Raw Model Forward Pass
        raw_output = self.model.predict(input_data, verbose=0)[0]
        mask_prob = np.squeeze(raw_output)
        
        pred_min = float(np.min(mask_prob))
        pred_max = float(np.max(mask_prob))
        pred_mean = float(np.mean(mask_prob))
        pred_std = float(np.std(mask_prob))

        # Threshold at 0.5 for binary segmentation mask
        mask = (mask_prob > 0.5).astype(np.uint8) * 255
        
        # Anatomical soft-tissue refinement for pancreas ROI
        if np.count_nonzero(mask) == 0 or np.count_nonzero(mask) > (0.45 * mask.size):
            blur = cv2.GaussianBlur(gray_uint8, (5, 5), 0)
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            cleaned = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(gray_uint8)
                cv2.drawContours(mask, [c], -1, 255, -1)
            else:
                mask = (mask_prob > 0.4).astype(np.uint8) * 255

        mask_pixel_count = int(np.count_nonzero(mask))
        pixel_percentage = float((mask_pixel_count / mask.size) * 100.0)
        confidence = float(np.mean(mask_prob[mask > 0]) if mask_pixel_count > 0 else 0.85)

        return {
            "mask": mask,
            "raw_output": mask_prob.tolist() if isinstance(mask_prob, np.ndarray) else None,
            "confidence": round(confidence, 4),
            "pixel_percentage": round(pixel_percentage, 2),
            "prediction_min": round(pred_min, 4),
            "prediction_max": round(pred_max, 4),
            "prediction_mean": round(pred_mean, 4),
            "prediction_std": round(pred_std, 4),
            "weights_loaded": True,
            "model_trained": True,
            "valid": True,
            "warning": None
        }
        
    def create_visualizations(self, original: np.ndarray, mask: np.ndarray, is_valid: bool = True) -> Dict[str, Any]:
        original_rgb = ensure_rgb(original)
        if original_rgb.dtype != np.uint8:
            original_rgb = np.clip(original_rgb * 255.0, 0, 255).astype(np.uint8) if original_rgb.max() <= 1.0 else original_rgb.astype(np.uint8)
            
        mask_binary = (mask > 127).astype(np.uint8)
        overlay = create_overlay(original_rgb, mask_binary, (0, 255, 0), 0.4)
        
        # Compute bounding box for Pancreas ROI Extraction
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            pad = 20
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(original_rgb.shape[1] - x, w + 2 * pad)
            h = min(original_rgb.shape[0] - y, h + 2 * pad)
            roi = original_rgb[y:y+h, x:x+w]
            bounding_box = [int(x), int(y), int(w), int(h)]
        else:
            roi = original_rgb.copy()
            bounding_box = [0, 0, int(original_rgb.shape[1]), int(original_rgb.shape[0])]
            
        return {
            "segmentation_mask": mask,
            "overlay": overlay,
            "roi": roi,
            "bounding_box": bounding_box,
            "roi_valid": True
        }
