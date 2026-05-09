import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# ----------------------------
# LOAD TRAINED MODEL
# ----------------------------
model = load_model("shape_model.keras")

# IMPORTANT: must match alphabetical order of dataset folders
class_labels = ["button", "checkbox", "radio", "textbox"]
         
# ----------------------------
# FOLDERS
# ----------------------------
input_folder = "input"
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)

images = os.listdir(input_folder)

# ----------------------------
# PROCESS IMAGES
# ----------------------------
for image_file in images:
    
    image_path = os.path.join(input_folder, image_file)
    print(f"\nProcessing: {image_file}")

    img = cv2.imread(image_path)

    if img is None:
        print("Invalid image. Skipping.")
        continue

    height, width = img.shape[:2]
    output = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print("No contours detected.")
        continue

    elements = []

    # ----------------------------
    # LOOP THROUGH CONTOURS
    # ----------------------------
    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w < 20 or h < 20:
            continue

        # ----------------------------
        # CROP SHAPE FOR CNN
        # ----------------------------
        crop = img[y:y+h, x:x+w]

        try:
            crop_resized = cv2.resize(crop, (128, 128))
        except:
            continue

        crop_resized = crop_resized / 255.0
        crop_resized = crop_resized.reshape(1, 128, 128, 3)

        prediction = model.predict(crop_resized, verbose=0)

        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        shape_name = class_labels[class_index]

        print(f"Detected: {shape_name} | Confidence: {confidence:.2f}")

        elements.append({
            "type": shape_name,
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })

        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(output, shape_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ----------------------------
    # SORT ELEMENTS
    # ----------------------------
    elements = sorted(elements, key=lambda el: (el["y"], el["x"]))

    # ----------------------------
    # GENERATE HTML
    # ----------------------------
    html_output = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Generated UI</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    body {{
        background: linear-gradient(135deg, #f6f8fd 0%, #f1f5f9 100%);
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 40px;
        min-height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    .canvas {{
        position: relative;
        width: {width}px;
        height: {height}px;
        background: rgba(255, 255, 255, 0.85);
        border-radius: 24px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }}
    /* Checkbox & Radio */
    .ctrl-input {{
        appearance: none;
        background-color: #fff;
        margin: 0;
        font: inherit;
        color: currentColor;
        width: 24px;
        height: 24px;
        border: 2px solid #cbd5e1;
        display: grid;
        place-content: center;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
    }}
    .ctrl-input:hover {{
        border-color: #6366f1;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
    }}
    .ctrl-input::before {{
        content: "";
        width: 12px;
        height: 12px;
        transform: scale(0);
        transition: 120ms transform ease-in-out;
        box-shadow: inset 1em 1em white;
    }}
    .ctrl-input:checked {{
        background-color: #6366f1;
        border-color: #6366f1;
    }}
    .ctrl-input:checked::before {{
        transform: scale(1);
    }}
    input[type="checkbox"].ctrl-input {{
        border-radius: 6px;
    }}
    input[type="checkbox"].ctrl-input::before {{
        transform-origin: bottom left;
        clip-path: polygon(14% 44%, 0 65%, 50% 100%, 100% 16%, 80% 0%, 43% 62%);
    }}
    input[type="radio"].ctrl-input {{
        border-radius: 50%;
    }}
    input[type="radio"].ctrl-input::before {{
        border-radius: 50%;
        background-color: white;
    }}
    /* Textbox */
    .textbox {{
        padding: 0.75rem 1rem;
        background: #f8fafc;
        border: 2px solid transparent;
        border-radius: 12px;
        font-size: 1rem;
        color: #1e293b;
        transition: all 0.3s ease;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.02);
        box-sizing: border-box;
    }}
    .textbox:focus {{
        outline: none;
        background: #fff;
        border-color: #6366f1;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
    }}
    .textbox::placeholder {{
        color: #94a3b8;
    }}
    /* Button */
    .btn {{
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .btn:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }}
    .btn:active {{
        transform: translateY(1px);
    }}
</style>
</head>
<body>
<div class="canvas">
"""

    for el in elements:

        left = el["x"]
        top = el["y"]

        if el["type"] == "checkbox":
            html_output += f"""
<input type="checkbox" class="ctrl-input" style="position:absolute; left:{left}px; top:{top}px;">
"""

        elif el["type"] == "radio":
            html_output += f"""
<input type="radio" class="ctrl-input" style="position:absolute; left:{left}px; top:{top}px;">
"""

        elif el["type"] == "textbox":
            # Using max to assure minimum reasonable width/height for a textbox
            tb_w = max(el['w'], 100)
            tb_h = max(el['h'], 40)
            html_output += f"""
<input type="text" class="textbox" placeholder="Input..." style="position:absolute; left:{left}px; top:{top}px; width:{tb_w}px; height:{tb_h}px;">
"""

        elif el["type"] == "button":
            btn_w = max(el['w'], 80)
            btn_h = max(el['h'], 40)
            html_output += f"""
<button class="btn" style="position:absolute; left:{left}px; top:{top}px; width:{btn_w}px; height:{btn_h}px;">
Action
</button>
"""

    html_output += """
</div>
</body>
</html>
"""

    name = os.path.splitext(image_file)[0]

    with open(os.path.join(output_folder, f"{name}.html"), "w") as f:
        f.write(html_output)

    cv2.imwrite(os.path.join(output_folder, f"{name}_detected.jpg"), output)

    print(f"{len(elements)} elements processed → HTML generated")