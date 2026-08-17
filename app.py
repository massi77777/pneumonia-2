"""
Streamlit app for detecting pneumonia from chest X-ray images.
Uses the trained model: pneumonia_model.keras
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = "pneumonia_model.keras"


# ------------------------------------------------------------------
# Core logic
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the model once and cache it in memory."""
    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(uploaded_image):
    """Prepare the uploaded image the same way the model was trained."""
    img = Image.open(uploaded_image).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def predict_image(model, img_array):
    """Predict the class and confidence score."""
    prob = model.predict(img_array)[0][0]
    if prob > 0.5:
        return "PNEUMONIA", float(prob)
    return "NORMAL", float(1 - prob)


# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        /* General */
        .stApp { background-color: #f7f9fb; }
        #MainMenu, footer, header { visibility: hidden; }

        .block-container { padding-top: 2.5rem; max-width: 760px; }

        /* Header */
        .app-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.25rem;
        }
        .app-header .icon {
            font-size: 2.1rem;
            line-height: 1;
        }
        .app-header h1 {
            font-size: 1.7rem;
            font-weight: 700;
            color: #10233e;
            margin: 0;
        }
        .app-subtitle {
            color: #5b6b7f;
            font-size: 0.98rem;
            margin-bottom: 1.75rem;
        }

        /* Card container */
        .card {
            background: #ffffff;
            border: 1px solid #e6ebf1;
            border-radius: 14px;
            padding: 1.6rem 1.6rem 1.4rem 1.6rem;
            box-shadow: 0 1px 3px rgba(16, 35, 62, 0.04);
            margin-bottom: 1.4rem;
        }
        .card h3 {
            font-size: 1rem;
            font-weight: 600;
            color: #10233e;
            margin-top: 0;
            margin-bottom: 0.9rem;
        }

        /* Uploader */
        [data-testid="stFileUploaderDropzone"] {
            background-color: #fafbfc;
            border: 1.5px dashed #c9d4e0;
            border-radius: 10px;
        }

        /* Result badges */
        .result-badge {
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
            letter-spacing: 0.02em;
        }
        .badge-pneumonia {
            background-color: #fdecec;
            color: #b3261e;
            border: 1px solid #f6c6c3;
        }
        .badge-normal {
            background-color: #e9f7ee;
            color: #1e7d38;
            border: 1px solid #bfe8cc;
        }

        .confidence-label {
            color: #5b6b7f;
            font-size: 0.88rem;
            margin-top: 0.9rem;
            margin-bottom: 0.25rem;
        }

        /* Disclaimer */
        .disclaimer {
            display: flex;
            gap: 0.6rem;
            align-items: flex-start;
            background-color: #fff8e6;
            border: 1px solid #f2e1a8;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            font-size: 0.85rem;
            color: #7a5c00;
            margin-top: 1.5rem;
        }

        .footer-note {
            text-align: center;
            color: #9aa7b6;
            font-size: 0.78rem;
            margin-top: 1.6rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <span class="icon">🩺</span>
        <h1>Pneumonia Detection from Chest X-Rays</h1>
    </div>
    <div class="app-subtitle">
        Upload a chest X-ray image and the model will assess whether it looks
        normal or shows signs of pneumonia.
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Upload card
# ------------------------------------------------------------------
st.markdown('<div class="card"><h3>1. Upload an X-ray image</h3>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Accepted formats: JPG, PNG",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Result card
# ------------------------------------------------------------------
if uploaded_file is not None:
    with st.spinner("Analyzing image..."):
        model = load_model()
        image, img_array = prepare_image(uploaded_file)
        result, confidence = predict_image(model, img_array)

    st.markdown('<div class="card"><h3>2. Result</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    with col2:
        badge_class = "badge-pneumonia" if result == "PNEUMONIA" else "badge-normal"
        label = "Pneumonia detected" if result == "PNEUMONIA" else "Normal"
        st.markdown(
            f'<span class="result-badge {badge_class}">{label}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="confidence-label">Confidence</div>',
            unsafe_allow_html=True,
        )
        st.progress(min(max(confidence, 0.0), 1.0))
        st.write(f"**{confidence:.1%}**")

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="card" style="text-align:center; color:#9aa7b6;">'
        "Your result will appear here once you upload an image."
        "</div>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# Disclaimer & footer
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer">
        <span>⚠️</span>
        <span>This tool is experimental and for educational purposes only.
        It is <b>not</b> a substitute for professional medical diagnosis.
        Always consult a qualified physician.</span>
    </div>
    <div class="footer-note">Powered by a MobileNetV2 deep learning model</div>
    """,
    unsafe_allow_html=True,
)
