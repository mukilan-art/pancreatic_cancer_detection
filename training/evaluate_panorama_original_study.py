
import os
import glob
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_PATH = "models/resnet50/panorama_resnet50.keras"
TEST_DIR = "../datasets/panorama_classification_clean/test"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

studies = defaultdict(lambda: {"files": [], "label": None})

for class_name, label in [("PDAC", 0), ("non-PDAC", 1)]:
    folder = os.path.join(TEST_DIR, class_name)

    for path in glob.glob(os.path.join(folder, "*.png")):
        filename = os.path.basename(path)
        study_id = filename.split("_slice_")[0]
        studies[study_id]["files"].append(path)
        studies[study_id]["label"] = label

correct = 0
total = 0
class_total = {"PDAC": 0, "non-PDAC": 0}
class_correct = {"PDAC": 0, "non-PDAC": 0}

print("=" * 60)
print("ORIGINAL RESNET50 - STUDY-LEVEL RESULTS")
print("=" * 60)

for study_id in sorted(studies):
    files = studies[study_id]["files"]
    true_label = studies[study_id]["label"]

    images = []

    for path in files:
        image = tf.keras.utils.load_img(
            path, target_size=(224, 224)
        )
        image = tf.keras.utils.img_to_array(image)
        image = preprocess_input(image)
        images.append(image)

    predictions = model.predict(
        np.array(images), verbose=0
    ).ravel()

    mean_non_pdac = float(np.mean(predictions))
    predicted_label = 1 if mean_non_pdac >= 0.5 else 0

    true_name = "PDAC" if true_label == 0 else "non-PDAC"
    predicted_name = "PDAC" if predicted_label == 0 else "non-PDAC"

    is_correct = predicted_label == true_label
    total += 1
    correct += int(is_correct)
    class_total[true_name] += 1
    class_correct[true_name] += int(is_correct)

    print(
        study_id,
        "| true =", true_name,
        "| predicted =", predicted_name,
        "| mean_nonPDAC =", round(mean_non_pdac, 4),
        "| slices =", len(files)
    )

print("=" * 60)
print("OVERALL ACCURACY:",
      round(correct / total * 100, 2), "%")
print("CORRECT STUDIES:", correct, "/", total)

for class_name in ["PDAC", "non-PDAC"]:
    n = class_total[class_name]
    c = class_correct[class_name]
    accuracy = (c / n * 100) if n else 0

    print(
        class_name, "ACCURACY:",
        round(accuracy, 2), "%",
        "| CORRECT:", c, "/", n
    )

print("=" * 60)
