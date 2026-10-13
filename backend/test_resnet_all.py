import cv2
import numpy as np
import tensorflow as tf
import glob
from tensorflow.keras.applications.resnet50 import preprocess_input

model = tf.keras.models.load_model(
    "models/resnet50/panorama_resnet50.keras"
)

base = "/mnt/f/pancreatic cancer detection/datasets/panorama_classification/test"

correct = 0
total = 0

pdac_correct = 0
pdac_total = 0

non_correct = 0
non_total = 0

for cls in ["PDAC", "non-PDAC"]:

    files = glob.glob(base + "/" + cls + "/*.png")

    for f in files:

        img = cv2.imread(f)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)

        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        output = float(model.predict(img, verbose=0)[0][0])

        pdac_probability = 1.0 - output

        if pdac_probability >= 0.5:
            prediction = "PDAC"
        else:
            prediction = "non-PDAC"

        total += 1

        if prediction == cls:
            correct += 1

        if cls == "PDAC":
            pdac_total += 1
            if prediction == "PDAC":
                pdac_correct += 1

        else:
            non_total += 1
            if prediction == "non-PDAC":
                non_correct += 1

print()
print("===== RESNET50 TEST RESULTS =====")
print("Total images:", total)
print("Correct:", correct)
print("Accuracy:", round(correct / total * 100, 2), "%")
print()
print("PDAC:", pdac_correct, "/", pdac_total,
      "=", round(pdac_correct / pdac_total * 100, 2), "%")
print()
print("non-PDAC:", non_correct, "/", non_total,
      "=", round(non_correct / non_total * 100, 2), "%")
