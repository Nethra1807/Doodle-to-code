import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# Load model
model = tf.keras.models.load_model("ui_model.keras")

# Try to load class names from `classes.txt` (one name per line).
# If not present, the script will print the raw prediction vector and top indices.
class_names = []
if os.path.exists("classes.txt"):
    with open("classes.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    preds = prediction[0]

    if len(class_names) == len(preds):
        predicted_class = class_names[np.argmax(preds)]
        confidence = np.max(preds)
        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")
    else:
        print("Model output length:", len(preds))
        print("Prediction vector:", preds)
        topk = np.argsort(preds)[::-1][:5]
        print("Top indices and probabilities:")
        for idx in topk:
            print(f"  Index {idx}: {preds[idx]:.4f}")
        print(f"Predicted index: {np.argmax(preds)}, confidence: {np.max(preds):.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    img_path = sys.argv[1]
    predict_image(img_path)