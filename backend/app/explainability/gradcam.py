import numpy as np
import tensorflow as tf
import cv2
from typing import Dict, Any, Optional


class GradCAM:
    def __init__(
        self,
        model: Optional[tf.keras.Model],
        last_conv_layer_name: str = "conv5_block3_out",
    ):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        self.ready = False
        self.last_conv_layer = None
        self.grad_model = None

        try:
            if model is None:
                raise ValueError("ResNet50 model is not loaded.")

            base_model = model.get_layer("resnet50")
            self.last_conv_layer = base_model.get_layer(
                last_conv_layer_name
            )

            feature_model = tf.keras.Model(
                inputs=base_model.inputs,
                outputs=self.last_conv_layer.output,
            )

            inputs = tf.keras.Input(
                shape=base_model.input_shape[1:]
            )
            features = feature_model(inputs)

            x = model.get_layer("global_average_pooling2d")(features)
            x = model.get_layer("dropout")(x)
            x = model.get_layer("dense")(x)
            x = model.get_layer("dropout_1")(x)
            predictions = model.get_layer("dense_1")(x)

            self.grad_model = tf.keras.Model(
                inputs=inputs,
                outputs=[features, predictions],
            )

            self.ready = True
            print("Grad-CAM initialized successfully.")

        except (AttributeError, ValueError, KeyError) as exc:
            self.ready = False
            print(f"Grad-CAM initialization failed: {exc}")

    def compute_gradcam(
        self,
        image: np.ndarray,
        pred_index: Optional[int] = None,
    ) -> Dict[str, Any]:

        if not self.ready or self.grad_model is None:
            return {
                "heatmap": None,
                "warning": "Grad-CAM model is not ready.",
            }

        img_array = image.copy()

        if img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0
        else:
            img_array = img_array.astype(np.float32)

        if img_array.ndim == 2:
            img_array = cv2.cvtColor(
                img_array, cv2.COLOR_GRAY2RGB
            )
        elif img_array.ndim == 3 and img_array.shape[-1] == 1:
            img_array = np.repeat(img_array, 3, axis=-1)

        if img_array.ndim != 3 or img_array.shape[-1] != 3:
            return {
                "heatmap": None,
                "warning": (
                    f"Expected a 3-channel image; got shape "
                    f"{img_array.shape}."
                ),
            }

        if img_array.shape[:2] != (224, 224):
            img_array = cv2.resize(
                img_array,
                (224, 224),
                interpolation=cv2.INTER_LINEAR,
            )

        img_array = np.expand_dims(img_array, axis=0)

        try:
            with tf.GradientTape() as tape:
                conv_outputs, predictions = self.grad_model(
                    img_array, training=False
                )

                # This model has one sigmoid output:
                # index 0 represents the single output neuron.
                class_channel = predictions[:, 0]

            gradients = tape.gradient(class_channel, conv_outputs)

            if gradients is None:
                return {
                    "heatmap": None,
                    "warning": (
                        "Grad-CAM could not calculate gradients "
                        "for the selected class."
                    ),
                }

            pooled_gradients = tf.reduce_mean(
                gradients, axis=(0, 1, 2)
            )
            conv_features = conv_outputs[0]

            heatmap = tf.reduce_sum(
                conv_features * pooled_gradients, axis=-1
            ).numpy()

        except Exception as exc:
            return {
                "heatmap": None,
                "warning": f"Grad-CAM computation failed: {exc}",
            }

        heatmap = np.maximum(heatmap, 0)
        max_cam = float(np.max(heatmap))

        if max_cam > 0:
            heatmap = heatmap / max_cam
        else:
            heatmap = np.zeros(
                heatmap.shape, dtype=np.float32
            )

        return {
            "heatmap": heatmap.astype(np.float32),
            "pred_index": 0,
            "confidence": float(predictions[0, 0]),
            "target_layer": self.last_conv_layer_name,
            "gradient_valid": True,
            "heatmap_min": float(np.min(heatmap)),
            "heatmap_max": float(np.max(heatmap)),
            "valid": max_cam > 0,
        }

    def create_heatmap_visualization(
        self,
        original_rgb: np.ndarray,
        heatmap: np.ndarray,
    ) -> Dict[str, np.ndarray]:

        if original_rgb.dtype != np.uint8:
            if original_rgb.max() <= 1.0:
                original_uint8 = np.clip(
                    original_rgb * 255.0, 0, 255
                ).astype(np.uint8)
            else:
                original_uint8 = np.clip(
                    original_rgb, 0, 255
                ).astype(np.uint8)
        else:
            original_uint8 = original_rgb.copy()

        heatmap_resized = cv2.resize(
            heatmap,
            (original_uint8.shape[1], original_uint8.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        heatmap_colored = np.uint8(
            255 * np.clip(heatmap_resized, 0, 1)
        )
        heatmap_colored = cv2.applyColorMap(
            heatmap_colored, cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            original_uint8, 0.55,
            heatmap_colored, 0.45,
            0,
        )

        return {
            "heatmap_colored": heatmap_colored,
            "overlay": overlay,
            "original": original_uint8,
        }
