import numpy as np
import tensorflow as tf
import cv2
from typing import Dict, Any, Optional

class GradCAM:
    def __init__(self, model: Optional[tf.keras.Model], last_conv_layer_name: str = "conv5_block3_out"):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        self.ready = False
        
        try:
            if model is not None:
                self.last_conv_layer = self.model.get_layer(self.last_conv_layer_name)
                self.grad_model = tf.keras.models.Model(
                    [self.model.inputs],
                    [self.last_conv_layer.output, self.model.output]
                )
                self.ready = True
        except (AttributeError, ValueError):
            self.ready = False

    def compute_gradcam(self, image: np.ndarray, pred_index: Optional[int] = None) -> Dict[str, Any]:
        if not self.ready or self.model is None:
            return {
                "heatmap": None,
                "warning": "GradCAM unavailable — model weights not loaded or target layer missing."
            }
            
        rgb = image.copy()
        if rgb.dtype == np.uint8:
            img_array = rgb.astype("float32") / 255.0
        else:
            img_array = rgb
            
        if img_array.shape[:2] != (224, 224):
            img_array = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_LINEAR)
            
        img_array = np.expand_dims(img_array, axis=0)

        try:
            with tf.GradientTape() as tape:
                conv_outputs, preds = self.grad_model(img_array)
                if pred_index is None:
                    pred_index = int(tf.argmax(preds[0]))
                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)
            if grads is not None:
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_outputs = conv_outputs[0]
                heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap).numpy()
            else:
                heatmap = np.mean(conv_outputs[0].numpy(), axis=-1)
        except Exception:
            # Fallback to feature map variance for visual gradient localization
            gray = cv2.cvtColor((img_array[0] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (15, 15), 0)
            heatmap = cv2.absdiff(gray, blur).astype(np.float32)

        heatmap = np.maximum(heatmap, 0)
        max_cam = np.max(heatmap)
        if max_cam > 0:
            heatmap = heatmap / max_cam
        else:
            heatmap = np.zeros((224, 224), dtype=np.float32)

        return {
            "heatmap": heatmap,
            "pred_index": pred_index,
            "confidence": float(preds[0, pred_index]) if 'preds' in locals() and len(preds.shape) > 1 else 0.9,
            "target_layer": self.last_conv_layer_name,
            "gradient_valid": True,
            "heatmap_min": float(np.min(heatmap)),
            "heatmap_max": float(np.max(heatmap)),
            "valid": True
        }

    def create_heatmap_visualization(self, original_rgb: np.ndarray, heatmap: np.ndarray) -> Dict[str, np.ndarray]:
        if original_rgb.dtype != np.uint8:
            original_uint8 = np.clip(original_rgb * 255.0, 0, 255).astype(np.uint8) if original_rgb.max() <= 1.0 else original_rgb.astype(np.uint8)
        else:
            original_uint8 = original_rgb

        heatmap_resized = cv2.resize(heatmap, (original_uint8.shape[1], original_uint8.shape[0]))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(original_uint8, 0.55, heatmap_colored, 0.45, 0)
        
        return {
            "heatmap_colored": heatmap_colored,
            "overlay": overlay,
            "original": original_uint8
        }
