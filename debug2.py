from PIL import Image, ImageDraw
from utils.preprocess import preprocess_image
from utils.predictor import UIComponentPredictor

predictor = UIComponentPredictor('model/ui_model.keras')

# Circle
img_circle = Image.new('RGB', (600, 450), 'white')
draw = ImageDraw.Draw(img_circle)
draw.ellipse((250, 175, 350, 275), outline='black', width=4)
proc_circle = preprocess_image(img_circle)
pred, conf = predictor.predict(proc_circle)
print(f"Circle image -> {pred} ({conf:.2f})")

# Rectangle
img_rect = Image.new('RGB', (600, 450), 'white')
draw = ImageDraw.Draw(img_rect)
draw.rectangle((200, 150, 400, 300), outline='black', width=4)
proc_rect = preprocess_image(img_rect)
pred, conf = predictor.predict(proc_rect)
print(f"Rect image -> {pred} ({conf:.2f})")

# Button
img_btn = Image.new('RGB', (600, 450), 'white')
draw = ImageDraw.Draw(img_btn)
draw.rectangle((200, 150, 400, 200), outline='black', width=4)
draw.text((250, 170), "Submit", fill="black")
proc_btn = preprocess_image(img_btn)
pred, conf = predictor.predict(proc_btn)
print(f"Button image -> {pred} ({conf:.2f})")
