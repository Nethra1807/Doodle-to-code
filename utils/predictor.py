import tensorflow as tf
import numpy as np
import os

class UIComponentPredictor:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = [
           "Button",
            "Radio",
            "checkbox_unchecked",
            "data_table",
            "radio_button_unchecked",
            "text_area"
        ]

    def predict(self, preprocessed_img):
        # Prediction
        prediction = self.model.predict(preprocessed_img)
        class_index = np.argmax(prediction)
        predicted_class = self.class_names[class_index]
        confidence = np.max(prediction)
        
        print(f"DEBUG: Prediction probabilities: {prediction}")
        print(f"DEBUG: Predicted index: {class_index}")
        print(f"DEBUG: Predicted class: {predicted_class}")
        print(f"DEBUG: Class names length: {len(self.class_names)}")
        
        return predicted_class, confidence
