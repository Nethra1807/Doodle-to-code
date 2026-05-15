import numpy as np
from PIL import Image, ImageDraw, ImageOps
from utils.predictor import UIComponentPredictor
import math

predictor = UIComponentPredictor('model/ui_model.keras')

def preprocess_better(img, target_size=(64, 64)):
    if isinstance(img, Image.Image):
        img_pil = img.convert('RGB')
    else:
        img_pil = Image.fromarray(img).convert('RGB')
        
    # Invert image to find bounding box of drawing
    # Assuming white background, black drawing
    inverted = ImageOps.invert(img_pil)
    # Convert to grayscale
    gray = inverted.convert("L")
    bbox = gray.getbbox()
    
    if bbox:
        # Crop to bbox
        img_cropped = img_pil.crop(bbox)
        
        # Make it square by padding
        w, h = img_cropped.size
        max_dim = max(w, h)
        
        # Create a new white square image
        new_img = Image.new('RGB', (max_dim, max_dim), 'white')
        
        # Paste the cropped image in the center
        paste_x = (max_dim - w) // 2
        paste_y = (max_dim - h) // 2
        new_img.paste(img_cropped, (paste_x, paste_y))
        
        # Add a little padding (10%)
        pad = int(max_dim * 0.1)
        padded_size = max_dim + 2 * pad
        final_img = Image.new('RGB', (padded_size, padded_size), 'white')
        final_img.paste(new_img, (pad, pad))
        
        # Resize to target
        img_resized = final_img.resize(target_size, Image.LANCZOS)
    else:
        # If blank, just resize
        img_resized = img_pil.resize(target_size, Image.LANCZOS)

    # Convert to numpy
    img_array = np.array(img_resized).astype('float32') / 255.0
    
    # Binarize (thresholding) to make lines solid black
    # Anything below 0.8 (fairly bright) becomes 0.0 (black)
    # Anything above becomes 1.0 (white)
    img_array = np.where(img_array < 0.9, 0.0, 1.0)
    
    return np.expand_dims(img_array, axis=0)


# Test with small circle
img_small = Image.new('RGB', (600, 450), 'white')
draw = ImageDraw.Draw(img_small)
draw.ellipse((280, 210, 320, 240), outline='black', width=4)

proc_small = preprocess_better(img_small)
pred, conf = predictor.predict(proc_small)
print(f"Small circle with BETTER preprocess -> {pred} ({conf:.2f})")
