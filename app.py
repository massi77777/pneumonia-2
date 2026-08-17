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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #0b0f17;
            --surface: #121826;
            --surface-2: #1a2436;
            --border: #2a374d;
            --border-soft: #212c40;
            --text-primary: #f7f9fc;
            --text-secondary: #b7c2d4;
            --text-muted: #8695ab;
            --accent: #4fd1c5;
            --accent-soft: rgba(79, 209, 197, 0.12);
            --danger: #ff6b6b;
            --danger-soft: rgba(255, 107, 107, 0.12);
            --danger-border: rgba(255, 107, 107, 0.35);
            --success: #37d67a;
            --success-soft: rgba(55, 214, 122, 0.12);
            --success-border: rgba(55, 214, 122, 0.35);
            --warn: #f5b942;
            --warn-soft: rgba(245, 185, 66, 0.1);
            --warn-border: rgba(245, 185, 66, 0.3);
        }

        html, body, .stApp {
            background-color: var(--bg) !important;
            font-family: 'Inter', -apple-system, sans-serif;
        }
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 780px; }

        * { color: var(--text-primary); }

        /* Header */
        .app-header {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            margin-bottom: 0.4rem;
        }
        .app-header .icon-wrap {
            width: 46px;
            height: 46px;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(79,209,197,0.18), rgba(79,209,197,0.04));
            border: 1px solid rgba(79,209,197,0.25);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
            flex-shrink: 0;
        }
        .app-header h1 {
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--text-primary) !important;
            margin: 0;
            letter-spacing: -0.01em;
        }
        .app-subtitle {
            color: var(--text-secondary) !important;
            font-size: 0.95rem;
            line-height: 1.5;
            margin: 0.6rem 0 2rem 0;
            max-width: 560px;
        }

        /* Section labels */
        .section-label {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--text-secondary) !important;
            margin-bottom: 0.85rem;
        }
        .section-label .num {
            width: 20px;
            height: 20px;
            border-radius: 5px;
            background: var(--accent-soft);
            border: 1px solid rgba(79,209,197,0.4);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            font-weight: 800;
            color: var(--accent) !important;
            flex-shrink: 0;
        }

        /* Card container */
        .card {
            background: var(--surface);
            border: 1px solid var(--border-soft);
            border-radius: 16px;
            padding: 1.6rem 1.7rem 1.5rem 1.7rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
            margin-bottom: 1.3rem;
        }

        /* Uploader */
        [data-testid="stFileUploaderDropzone"] {
            background-color: var(--surface-2) !important;
            border: 1.5px dashed var(--border) !important;
            border-radius: 12px !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--accent) !important;
        }
        [data-testid="stFileUploader"] section { color: var(--text-secondary) !important; }
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] small {
            color: var(--text-secondary) !important;
        }
        [data-testid="stBaseButton-secondary"] {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            color: var(--text-primary) !important;
        }
        [data-testid="stFileUploaderFile"] {
            background-color: var(--surface-2) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px;
            padding: 0.5rem 0.7rem;
        }
        [data-testid="stFileUploaderFile"] * {
            color: var(--text-primary) !important;
        }
        [data-testid="stFileUploaderFileName"] {
            color: var(--text-primary) !important;
            font-weight: 500;
        }
        [data-testid="stFileUploaderFile"] small,
        [data-testid="stFileUploaderFile"] span[class*="fileSize"] {
            color: var(--text-muted) !important;
        }
        [data-testid="stFileUploaderFileIcon"] svg,
        [data-testid="stFileUploaderFile"] svg {
            fill: var(--accent) !important;
            color: var(--accent) !important;
        }
        [data-testid="stFileUploaderFile"] [data-testid="stBaseButton-minimal"] {
            color: var(--text-secondary) !important;
        }
        [data-testid="baseButton-minimal"] svg { fill: var(--text-secondary) !important; }
        [data-testid="stFileUploaderDeleteBtn"] button {
            color: var(--text-secondary) !important;
        }

        /* Result badges */
        .result-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.4rem 1rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.92rem;
            letter-spacing: 0.01em;
        }
        .badge-pneumonia {
            background-color: var(--danger-soft);
            color: var(--danger) !important;
            border: 1px solid var(--danger-border);
        }
        .badge-pneumonia * { color: var(--danger) !important; }
        .badge-normal {
            background-color: var(--success-soft);
            color: var(--success) !important;
            border: 1px solid var(--success-border);
        }
        .badge-normal * { color: var(--success) !important; }

        .confidence-label {
            color: var(--text-muted) !important;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-top: 1.3rem;
            margin-bottom: 0.5rem;
        }
        .confidence-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text-primary) !important;
            margin-top: 0.4rem;
        }

        /* Progress bar */
        div[data-testid="stProgress"] > div > div {
            background-color: var(--surface-2) !important;
            border-radius: 999px;
        }
        div[data-testid="stProgress"] > div > div > div {
            background: linear-gradient(90deg, var(--accent), #38b2ac) !important;
            border-radius: 999px;
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            color: var(--text-muted) !important;
            padding: 2.2rem 1rem;
            font-size: 0.9rem;
        }
        .empty-state .emoji { font-size: 1.6rem; display: block; margin-bottom: 0.6rem; opacity: 0.6; }

        /* Disclaimer */
        .disclaimer {
            display: flex;
            gap: 0.65rem;
            align-items: flex-start;
            background-color: var(--warn-soft);
            border: 1px solid var(--warn-border);
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            font-size: 0.83rem;
            line-height: 1.5;
            color: #e0c479 !important;
            margin-top: 1.6rem;
        }
        .disclaimer * { color: #e0c479 !important; }
        .disclaimer b { color: #f5d98a !important; }

        .footer-note {
            text-align: center;
            color: var(--text-muted) !important;
            font-size: 0.76rem;
            margin-top: 1.6rem;
            letter-spacing: 0.02em;
        }

        img { border-radius: 10px; }
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
        <div class="icon-wrap">🩺</div>
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
st.markdown(
    '<div class="card">'
    '<div class="section-label"><span class="num">1</span> Upload X-ray image</div>',
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader(
    "Accepted formats: JPG, PNG",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
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

    st.markdown(
        '<div class="card">'
        '<div class="section-label"><span class="num">2</span> Analysis result</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    with col2:
        badge_class = "badge-pneumonia" if result == "PNEUMONIA" else "badge-normal"
        dot = "●"
        label = "Pneumonia detected" if result == "PNEUMONIA" else "Normal"
        st.markdown(
            f'<span class="result-badge {badge_class}">{dot} {label}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="confidence-label">Model confidence</div>',
            unsafe_allow_html=True,
        )
        st.progress(min(max(confidence, 0.0), 1.0))
        st.markdown(
            f'<div class="confidence-value">{confidence:.1%}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="card">'
        '<div class="empty-state">'
        '<span class="emoji">📤</span>'
        "Your result will appear here once you upload an image."
        "</div></div>",
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
