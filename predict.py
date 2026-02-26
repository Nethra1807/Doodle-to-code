import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# Load model
model = tf.keras.models.load_model("ui_model.keras")

# Class labels (IMPORTANT: Must match printed class_indices from training)
class_names = ['Button', 'Checkbox', 'Form', 'Radio', 'Table']  # Adjust if needed

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    img_path = sys.argv[1]
    predict_image(img_path)