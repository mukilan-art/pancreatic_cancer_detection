import numpy as np
import cv2
from typing import Dict, Any, List
from tensorflow.keras.applications.resnet50 import preprocess_input

from app.utils.image_utils import ensure_rgb


class ResNet50Inference:

    def __init__(
        self,
        model: Any,
        weights_loaded: bool,
        classes: List[str],
        model_path: str = ""
    ):
        self.model = model
        self.weights_loaded = weights_loaded
        self.classes = classes
        self.model_path = model_path
        self.model_trained = True

    def classify(self, image: np.ndarray) -> Dict[str, Any]:

        if not self.weights_loaded or self.model is None:
            return {
                "prediction": None,
                "confidence": None,
                "probabilities": None,
                "raw_preds": None,
                "weights_loaded": False,
                "model_trained": False,
                "valid": False,
                "warning": "ResNet50 model weights unavailable."
            }

        # ----------------------------------------------------
        # Convert image to RGB
        # ----------------------------------------------------

        rgb_image = ensure_rgb(image)

        # ----------------------------------------------------
        # Resize to ResNet50 input size
        # ----------------------------------------------------

        if rgb_image.shape[:2] != (224, 224):
            rgb_image = cv2.resize(
                rgb_image,
                (224, 224),
                interpolation=cv2.INTER_LINEAR
            )

        # ----------------------------------------------------
        # Convert to float32
        # ----------------------------------------------------

        input_data = rgb_image.astype(np.float32)

        # ----------------------------------------------------
        # PANORAMA ResNet50 preprocessing
        # Same preprocessing used during training
        # ----------------------------------------------------

        input_data = preprocess_input(input_data)

        # Add batch dimension
        input_data = np.expand_dims(
            input_data,
            axis=0
        )

        # ----------------------------------------------------
        # Model prediction
        #
        # Model output:
        # sigmoid probability
        # 0 = PDAC
        # 1 = non-PDAC
        # ----------------------------------------------------

        raw_prediction = self.model.predict(
            input_data,
            verbose=0
        )

        probability_non_pdac = float(
            raw_prediction[0][0]
        )

        probability_pdac = (
            1.0 - probability_non_pdac
        )

        # ----------------------------------------------------
        # Classification threshold
        # ----------------------------------------------------

        if probability_non_pdac >= 0.5:

            prediction = "non-PDAC"
            confidence = probability_non_pdac

        else:

            prediction = "PDAC"
            confidence = probability_pdac

        probabilities = {
            "PDAC": probability_pdac,
            "non-PDAC": probability_non_pdac
        }

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "raw_preds": [
                probability_pdac,
                probability_non_pdac
            ],
            "weights_loaded": True,
            "model_trained": True,
            "valid": True,
            "warning": None
        }
