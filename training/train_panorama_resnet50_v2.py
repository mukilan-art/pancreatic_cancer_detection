import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ============================================================
# PANORAMA ResNet50 V2
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "datasets", "panorama_classification_clean")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "models", "resnet50")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "panorama_resnet50_v2.keras")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

print("=" * 60)
print("PANORAMA ResNet50 V2 TRAINING")
print("=" * 60)
print("Dataset:", DATA_DIR)
print("Output :", MODEL_PATH)
print("=" * 60)

# ------------------------------------------------------------
# GPU
# ------------------------------------------------------------

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:", gpus)
else:
    print("WARNING: No GPU detected. Training will use CPU.")

# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

train_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_names=["PDAC", "non-PDAC"],
    seed=SEED
)

val_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Class names:", train_ds.class_names)

# ------------------------------------------------------------
# Convert to float32 and ResNet50 preprocessing
# ------------------------------------------------------------

def preprocess_dataset(images, labels):
    images = tf.cast(images, tf.float32)
    images = preprocess_input(images)
    return images, labels


train_ds = train_ds.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------------------
# Calculate class weights
# ------------------------------------------------------------

# Explicit class mapping:
# PDAC = 0
# non-PDAC = 1

class_counts = {
    0: len(os.listdir(os.path.join(DATA_DIR, "train", "PDAC"))),
    1: len(os.listdir(os.path.join(DATA_DIR, "train", "non-PDAC")))
}

total = sum(class_counts.values())

class_weight = {
    0: total / (2.0 * class_counts[0]),
    1: total / (2.0 * class_counts[1])
}

print("Class counts:", class_counts)
print("Class weights:", class_weight)

# ------------------------------------------------------------
# Data augmentation
# ------------------------------------------------------------

augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.05, 0.05),
    ],
    name="augmentation"
)

# ------------------------------------------------------------
# ResNet50 base
# ------------------------------------------------------------

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# ------------------------------------------------------------
# Classification head
# ------------------------------------------------------------

inputs = keras.Input(shape=(224, 224, 3))

x = augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = keras.Model(inputs, outputs)

model.summary()

# ------------------------------------------------------------
# Phase 1: Train classification head
# ------------------------------------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc")
    ]
)

callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

print("\n" + "=" * 60)
print("PHASE 1: TRAINING CLASSIFICATION HEAD")
print("=" * 60)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weight,
    callbacks=callbacks
)

# ------------------------------------------------------------
# Phase 2: Fine-tune upper ResNet50 layers
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PHASE 2: FINE-TUNING UPPER RESNET50 LAYERS")
print("=" * 60)

base_model.trainable = True

# Freeze everything except the final ~30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc")
    ]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weight,
    callbacks=callbacks
)

# ------------------------------------------------------------
# Load best model
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)

best_model = keras.models.load_model(MODEL_PATH)

# ------------------------------------------------------------
# Final test evaluation
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

results = best_model.evaluate(
    test_ds,
    verbose=1
)

for name, value in zip(best_model.metrics_names, results):
    print(f"{name}: {value:.4f}")

print("\nTraining complete.")
print("Best model saved to:")
print(MODEL_PATH)
