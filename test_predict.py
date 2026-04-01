import numpy as np
from PIL import Image
from utils.preprocess import preprocess_image
from utils.predictor import UIComponentPredictor

print("Creating dummy image simulating canvas output (RGBA)...")
# Canvas gives RGBA image, let's say 400x400
dummy_canvas_data = np.zeros((400, 400, 4), dtype=np.uint8)
# Add some white background
dummy_canvas_data[:, :, :3] = 255
dummy_canvas_data[:, :, 3] = 255
# Draw a simple black shape (button-like)
dummy_canvas_data[150:250, 100:300, :3] = 0

print("Preprocessing image...")
processed_img = preprocess_image(dummy_canvas_data)

print(f"Processed image shape: {processed_img.shape}")

print("Loading predictor...")
predictor = UIComponentPredictor("model/ui_model.keras")

print("Running prediction...")
try:
    label, confidence = predictor.predict(processed_img)
    print(f"\nSUCCESS! Predicted: {label} with {confidence*100:.2f}% confidence.")
except Exception as e:
    import traceback
    print("\nFAILED with exception:")
    traceback.print_exc()
