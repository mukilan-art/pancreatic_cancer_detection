import os
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

def build_resnet50(num_classes: int = 2) -> Model:
    base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def load_resnet50(model_path: str) -> Tuple[Optional[Model], bool]:
    if not os.path.exists(model_path):
        return None, False

    try:
        model = tf.keras.models.load_model(model_path)
        return model, True
    except Exception as e:
        print(f"Error loading ResNet50 model from {model_path}: {e}")
        return None, False
