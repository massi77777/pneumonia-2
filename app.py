"""
Pneumonia Detection App - Streamlit
Loads pneumonia_model.keras and classifies chest X-ray images.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = "pneumonia_model.keras"

st.set_page_config(page_title="Pneumonia Detector", page_icon="🩺", layout="centered")

st.markdown(
    """
    <style>
    .stApp{ background-color:#0d0d0d; color:#f5f5f5; }
    h1{
        font-size:3rem !important;
        font-weight:900 !important;
        text-align:center;
        background: linear-gradient(90deg,#00ffcc,#ff00c8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom:0;
    }
    .subtitle{
        text-align:center;
        font-size:1.2rem;
        color:#aaaaaa;
        margin-bottom:30px;
    }
    [data-testid="stFileUploaderDropzone"]{
        background-color:#1a1a1a;
        border:3px dashed #00ffcc !important;
        border-radius:16px;
    }
    .result{
        text-align:center;
        border-radius:20px;
        padding:30px;
        margin-top:20px;
        font-weight:900;
        font-size:2.3rem;
    }
    .pneumonia{
        background-color:#2a0010;
        color:#ff003c;
        border:3px solid #ff003c;
        text-shadow: 0 0 12px #ff003c;
    }
    .normal{
        background-color:#00220f;
        color:#00ff88;
        border:3px solid #00ff88;
        text-shadow: 0 0 12px #00ff88;
    }
    .confidence{
        text-align:center;
        font-size:1.5rem;
        font-weight:800;
        color:#ffd400;
        margin-top:10px;
    }
    .stProgress > div > div{ background-color:#ffd400 !important; }
    footer{ visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def loadModel():
    return tf.keras.models.load_model(MODEL_PATH)


def prepareImage(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    return img, img_array


def predictImage(model, img_array):
    prob = model.predict(img_array)[0][0]
    if prob > 0.5:
        return "PNEUMONIA ⚠️", prob, "pneumonia"
    return "NORMAL ✅", 1 - prob, "normal"


st.markdown("<h1>🩺 PNEUMONIA AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a chest X-ray. Get an instant AI verdict.</p>", unsafe_allow_html=True)

model = loadModel()
uploaded_file = st.file_uploader("Upload X-ray (jpg/png)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    with st.spinner("Analyzing..."):
        image, img_array = prepareImage(uploaded_file)
        result, confidence, style = predictImage(model, img_array)

    st.image(image, use_container_width=True)
    st.markdown(f"<div class='result {style}'>{result}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='confidence'>{confidence:.0%} CONFIDENCE</div>", unsafe_allow_html=True)
    st.progress(float(confidence))

st.markdown("---")
st.caption("⚠️ Experimental tool. Not a substitute for professional medical diagnosis.")
