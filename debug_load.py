import tensorflow as tf
import os
import traceback

model_path = "model/ui_model.keras"

print(f"Checking {model_path}...")
print(f"File exists: {os.path.exists(model_path)}")
if os.path.exists(model_path):
    print(f"File size: {os.path.getsize(model_path)} bytes")

try:
    print("Attempting to load model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    model.summary()
except Exception as e:
    print("Error loading model:")
    print(str(e))
    traceback.print_exc()
