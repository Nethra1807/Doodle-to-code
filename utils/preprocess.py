import cv2
import numpy as np
from PIL import Image

def preprocess_image(img, target_size=(64, 64)):
    """
    Preprocess the input image for CNN prediction.
    - Convert to RGB (model expects RGB based on predict.py)
    - Resize to target size
    - Normalize to [0, 1]
    - Expand dims for batch processing
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(img, Image.Image):
        img = np.array(img.convert('RGB'))
    
    # If image is RGBA (from canvas), convert to RGB
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize
    img_resized = cv2.resize(img, target_size)
    
    # Normalize
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Add batch dimension
    img_final = np.expand_dims(img_normalized, axis=0)
    
    return img_final
