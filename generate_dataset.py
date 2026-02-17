import cv2
import numpy as np
import os
import random

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = 200
IMAGES_PER_CLASS = 50

CLASSES = ["button", "checkbox", "radio", "textbox"]

# Create folders
for cls in CLASSES:
    os.makedirs(f"dataset/{cls}", exist_ok=True)

# ----------------------------
# GENERATE IMAGES
# ----------------------------
for cls in CLASSES:
    for i in range(IMAGES_PER_CLASS):

        img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

        thickness = random.randint(2, 6)

        if cls == "checkbox":
            size = random.randint(50, 100)
            x = random.randint(20, 80)
            y = random.randint(20, 80)

            cv2.rectangle(
                img,
                (x, y),
                (x + size, y + size),
                (0, 0, 0),
                thickness
            )

        elif cls == "radio":
            radius = random.randint(25, 50)
            x = random.randint(60, 140)
            y = random.randint(60, 140)

            cv2.circle(
                img,
                (x, y),
                radius,
                (0, 0, 0),
                thickness
            )

        elif cls == "button":
            width = random.randint(80, 140)
            height = random.randint(40, 70)
            x = random.randint(20, 60)
            y = random.randint(60, 100)

            cv2.rectangle(
                img,
                (x, y),
                (x + width, y + height),
                (0, 0, 0),
                thickness
            )

        elif cls == "textbox":
            width = random.randint(120, 170)
            height = random.randint(30, 50)
            x = random.randint(10, 40)
            y = random.randint(80, 110)

            cv2.rectangle(
                img,
                (x, y),
                (x + width, y + height),
                (0, 0, 0),
                thickness
            )

        cv2.imwrite(f"dataset/{cls}/{cls}_{i}.png", img)

print("Dataset generation complete.")
