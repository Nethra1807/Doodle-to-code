import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# Load model
model = tf.keras.models.load_model("model/ui_model.keras")

# Class labels (IMPORTANT: Must match printed class_indices from training)
class_names = [
"Button",
"Radio",
"checkbox_unchecked",
"data_table",
"radio_button_unchecked",
"text_area"
]

def predict_image(img_path):
    # Preprocessing
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_class = class_names[class_index]
    confidence = np.max(prediction)

    print(f"DEBUG: Prediction probabilities: {prediction}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    img_path = sys.argv[1]
    predict_image(img_path)