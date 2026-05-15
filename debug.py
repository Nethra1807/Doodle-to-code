from PIL import Image, ImageDraw
import numpy as np
from utils.preprocess import preprocess_image
from utils.predictor import UIComponentPredictor

predictor = UIComponentPredictor('model/ui_model.keras')

# 1. Blank white image
img_blank = Image.new('RGB', (600, 450), 'white')
proc_blank = preprocess_image(img_blank)
pred, conf = predictor.predict(proc_blank)
print(f"Blank image -> {pred} ({conf:.2f})")

# 2. Thin line (like frontend)
img_thin = Image.new('RGB', (600, 450), 'white')
draw = ImageDraw.Draw(img_thin)
draw.line((100, 100, 500, 350), fill='black', width=4)
proc_thin = preprocess_image(img_thin)
pred, conf = predictor.predict(proc_thin)
print(f"Thin line image -> {pred} ({conf:.2f})")

# 3. Thick line
img_thick = Image.new('RGB', (600, 450), 'white')
draw = ImageDraw.Draw(img_thick)
draw.line((100, 100, 500, 350), fill='black', width=20)
proc_thick = preprocess_image(img_thick)
pred, conf = predictor.predict(proc_thick)
print(f"Thick line image -> {pred} ({conf:.2f})")
