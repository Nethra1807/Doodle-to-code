
from flask import Flask, request, redirect, url_for, render_template_string
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model at startup
MODEL_PATH = "ui_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names if available
CLASS_FILE = "classes.txt"
class_names = []
if os.path.exists(CLASS_FILE):
    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        class_names = [l.strip() for l in f if l.strip()]

INDEX_HTML = """
<!doctype html>
<title>Doodle-to-Code - Predict</title>
<h1>Upload a sketch image</h1>
<form method=post enctype=multipart/form-data action="/predict">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if result %}
  <h2>Result</h2>
  <pre>{{ result }}</pre>
{% endif %}
"""


def predict_image_bytes(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML, result=None)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    f = request.files['file']
    if f.filename == '':
        return redirect(url_for('index'))

    save_path = os.path.join('tmp_upload.png')
    f.save(save_path)

    preds = predict_image_bytes(save_path)
    topk = np.argsort(preds)[::-1][:5]

    lines = []
    lines.append(f"Model output length: {len(preds)}")
    if len(class_names) == len(preds):
        idx = int(np.argmax(preds))
        lines.append(f"Prediction: {class_names[idx]} (index {idx})")
        lines.append(f"Confidence: {preds[idx]:.4f}")
    else:
        lines.append("Top indices and probabilities:")
        for i in topk:
            lines.append(f"Index {int(i)}: {preds[int(i)]:.4f}")

    result = "\n".join(lines)
    return render_template_string(INDEX_HTML, result=result)


if __name__ == '__main__':
    # Run on localhost:5000
    app.run(host='127.0.0.1', port=5000)
