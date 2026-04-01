import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import os

from utils.preprocess import preprocess_image
from utils.predictor import UIComponentPredictor
from utils.html_mapper import map_class_to_html

# Page Config
st.set_page_config(page_title="Doodle-to-Code Converter", layout="wide", page_icon="🎨")

st.title("🎨 Doodle-to-Code Converter")
st.markdown("""
Convert your UI sketches into HTML code instantly! 
Draw on the canvas below or upload an image of your sketch.
""")

import traceback

# Initialize Predictor with caching
@st.cache_resource
def load_predictor(model_path):
    return UIComponentPredictor(model_path)

try:
    predictor = load_predictor("model/ui_model.keras")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.text(traceback.format_exc())
    st.stop()

# Sidebar for controls
st.sidebar.title("Controls")
drawing_mode = st.sidebar.selectbox("Drawing Tool:", ("freedraw", "line", "rect", "circle", "transform"))
stroke_width = st.sidebar.slider("Stroke Width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke Color: ", "#000000")
bg_color = st.sidebar.color_picker("Background Color: ", "#ffffff")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Sketch your UI Component")
    
    # Selection for Input Type
    input_type = st.radio("Choose Input Method:", ("Draw on Canvas", "Upload Image"))
    
    input_image = None
    
    if input_type == "Draw on Canvas":
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=400,
            width=400,
            drawing_mode=drawing_mode,
            key="canvas",
        )
        if canvas_result.image_data is not None:
            input_image = canvas_result.image_data
            
    else:
        uploaded_file = st.file_uploader("Upload a sketch image...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Uploaded Image", use_container_width=False, width=400)

    generate_button = st.button("🚀 Generate Code", type="primary")

with col2:
    st.subheader("Result & HTML Code")
    
    if generate_button and input_image is not None:
        with st.spinner("Analyzing sketch..."):
            # Preprocess
            processed_img = preprocess_image(input_image)
            
            # Predict
            label, confidence = predictor.predict(processed_img)
            
            # Map to HTML
            html_code = map_class_to_html(label)
            
            # Display Prediction
            st.success(f"**Detected Component:** {label} ({confidence*100:.2f}% Confidence)")
            
            # Display Code
            st.markdown("### 📋 Generated HTML")
            st.code(html_code, language="html")
            
            # Preview
            st.markdown("### 👁️ Component Preview")
            st.components.v1.html(html_code, height=200, scrolling=True)
            
            # Download button
            st.download_button(
                label="📥 Download HTML",
                data=html_code,
                file_name=f"{label.lower()}_snippet.html",
                mime="text/html"
            )
    elif generate_button:
        st.warning("Please draw something or upload an image first!")
    else:
        st.info("Sketch something and click 'Generate Code' to see the magic!")

st.markdown("---")
st.caption("Built with Streamlit, TensorFlow, and OpenCV.")
