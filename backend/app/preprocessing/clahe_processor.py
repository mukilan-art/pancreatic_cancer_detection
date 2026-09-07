import cv2
import numpy as np
from typing import Dict

class CLAHEProcessor:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
        self.clip_limit = clip_limit
        self.tile_grid_size = (tile_grid_size, tile_grid_size)
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            cl = self.clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        else:
            return self.clahe.apply(image)

    def apply_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (3, 3), 0)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        return (image.astype(np.float32) / 255.0)

    def preprocess(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        denoised = self.apply_noise_reduction(image)
        clahe_enhanced = self.apply_clahe(denoised)
        normalized = self.normalize(clahe_enhanced)
        
        resized_224 = cv2.resize(clahe_enhanced, (224, 224), interpolation=cv2.INTER_AREA)
        resized_512 = cv2.resize(clahe_enhanced, (512, 512), interpolation=cv2.INTER_AREA)
        
        return {
            "original": image,
            "denoised": denoised,
            "clahe_enhanced": clahe_enhanced,
            "normalized": normalized,
            "resized_224": resized_224,
            "resized_512": resized_512
        }
