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
<html>
<body style="background:#f2f2f2; font-family:Arial;">
<div style="position:relative; width:{width}px; height:{height}px;">
"""

    for el in elements:

        left = el["x"]
        top = el["y"]

        if el["type"] == "checkbox":
            html_output += f"""
<input type="checkbox" style="
position:absolute;
left:{left}px;
top:{top}px;">
"""

        elif el["type"] == "radio":
            html_output += f"""
<input type="radio" style="
position:absolute;
left:{left}px;
top:{top}px;">
"""

        elif el["type"] == "textbox":
            html_output += f"""
<input type="text" style="
position:absolute;
left:{left}px;
top:{top}px;
width:{el['w']}px;
height:32px;
padding:4px;
border:1px solid #444;">
"""

        elif el["type"] == "button":
            html_output += f"""
<button style="
position:absolute;
left:{left}px;
top:{top}px;
width:{el['w']}px;
height:40px;
background-color:#1976d2;
color:white;
border:none;
border-radius:6px;
cursor:pointer;">
Button
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
