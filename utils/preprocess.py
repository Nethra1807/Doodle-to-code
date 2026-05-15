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
        img_pil = img.convert('RGB')
    else:
        # numpy array input
        img_pil = Image.fromarray(img)
        if img_pil.mode == 'RGBA':
            img_pil = img_pil.convert('RGB')
        elif img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')

    # Resize
    img_resized = img_pil.resize(target_size, Image.LANCZOS)

    # Convert to numpy
    img_array = np.array(img_resized).astype('float32') / 255.0

    # Add batch dimension
    img_final = np.expand_dims(img_array, axis=0)

    return img_final
