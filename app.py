"""
Pneumonia Detection AI - Streamlit
Modern Medical AI Dashboard
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
MODEL_PATH = "pneumonia_model.keras"


st.set_page_config(
    page_title="Pneumonia AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(0, 255, 204, 0.08),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(100, 80, 255, 0.08),
                transparent 30%
            ),
            #080b10;

        color: #ffffff;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 35px;
        padding-bottom: 50px;
    }


    /* =====================================================
       HEADER
       ===================================================== */

    .header {
        text-align: center;
        padding: 20px 0 35px 0;
    }

    .logo {
        font-size: 55px;
        margin-bottom: 5px;
    }

    .title {
        font-size: 48px;
        font-weight: 900;
        letter-spacing: 2px;
        margin: 0;
        background: linear-gradient(
            90deg,
            #00ffcc,
            #00d9ff,
            #9b7cff
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        color: #9ba7b7;
        font-size: 18px;
        margin-top: 8px;
    }


    /* =====================================================
       GLASS CARD
       ===================================================== */

    .card {
        background: rgba(18, 23, 31, 0.75);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 28px;
        box-shadow:
            0 20px 50px rgba(0,0,0,0.35),
            inset 0 1px 0 rgba(255,255,255,0.04);

        backdrop-filter: blur(15px);
    }


    /* =====================================================
       SECTION TITLE
       ===================================================== */

    .section-title {
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 6px;
        color: #ffffff;
    }

    .section-description {
        color: #8995a5;
        font-size: 14px;
        margin-bottom: 20px;
    }


    /* =====================================================
       UPLOAD AREA
       ===================================================== */

    [data-testid="stFileUploaderDropzone"] {

        background:
            linear-gradient(
                145deg,
                rgba(0,255,204,0.06),
                rgba(90,80,255,0.05)
            ) !important;

        border: 2px dashed #00d9b0 !important;

        border-radius: 18px !important;

        padding: 30px !important;

        transition: all 0.3s ease;
    }

    [data-testid="stFileUploaderDropzone"]:hover {

        border-color: #00ffcc !important;

        box-shadow:
            0 0 25px rgba(0,255,204,0.15);

        transform: translateY(-2px);
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;
    }


    [data-testid="stFileUploaderDropzone"] button {

        background: #00e6b8 !important;

        color: #07110f !important;

        border: none !important;

        border-radius: 10px !important;

        font-weight: 800 !important;

        padding: 8px 18px !important;

        box-shadow:
            0 0 18px rgba(0,255,204,0.25);
    }

    [data-testid="stFileUploaderDropzone"] button:hover {

        background: #00ffcc !important;
    }


    /* =====================================================
       IMAGE
       ===================================================== */

    [data-testid="stImage"] img {

        border-radius: 18px;

        border: 1px solid rgba(255,255,255,0.08);

        box-shadow:
            0 15px 40px rgba(0,0,0,0.4);
    }


    /* =====================================================
       RESULT CARDS
       ===================================================== */

    .result-card {

        padding: 35px 25px;

        border-radius: 22px;

        text-align: center;

        margin-top: 20px;

        position: relative;

        overflow: hidden;
    }


    .result-card::before {

        content: "";

        position: absolute;

        width: 180px;
        height: 180px;

        border-radius: 50%;

        top: -90px;
        right: -60px;

        opacity: 0.15;
    }


    .result-icon {

        font-size: 55px;

        margin-bottom: 5px;
    }


    .result-title {

        font-size: 34px;

        font-weight: 900;

        letter-spacing: 1px;

        margin-bottom: 8px;
    }


    .result-description {

        font-size: 15px;

        color: #aab4c2;

        margin-bottom: 20px;
    }


    /* PNEUMONIA */

    .pneumonia-card {

        background:
            linear-gradient(
                145deg,
                rgba(255,0,60,0.15),
                rgba(50,5,20,0.8)
            );

        border: 1px solid rgba(255,0,60,0.5);

        box-shadow:
            0 0 35px rgba(255,0,60,0.08);
    }


    .pneumonia-title {

        color: #ff3158;

        text-shadow:
            0 0 15px rgba(255,0,60,0.5);
    }


    /* NORMAL */

    .normal-card {

        background:
            linear-gradient(
                145deg,
                rgba(0,255,136,0.12),
                rgba(0,35,20,0.8)
            );

        border: 1px solid rgba(0,255,136,0.45);

        box-shadow:
            0 0 35px rgba(0,255,136,0.07);
    }


    .normal-title {

        color: #00ff88;

        text-shadow:
            0 0 15px rgba(0,255,136,0.5);
    }


    /* =====================================================
       CONFIDENCE
       ===================================================== */

    .confidence-label {

        display: flex;

        justify-content: space-between;

        margin-top: 25px;

        margin-bottom: 8px;

        color: #9ba7b7;

        font-size: 14px;

        font-weight: 700;
    }


    .confidence-value {

        color: #ffffff;

        font-weight: 900;
    }


    /* Streamlit progress */

    .stProgress > div > div {

        background-color: #00e6b8 !important;

        border-radius: 20px;
    }

    .stProgress {

        height: 10px;
    }


    /* =====================================================
       INFO BOXES
       ===================================================== */

    .info-box {

        background: rgba(255,255,255,0.035);

        border: 1px solid rgba(255,255,255,0.07);

        border-radius: 14px;

        padding: 15px;

        margin-top: 10px;
    }

    .info-label {

        color: #7f8b9a;

        font-size: 12px;

        text-transform: uppercase;

        letter-spacing: 1px;
    }

    .info-value {

        color: #ffffff;

        font-size: 16px;

        font-weight: 800;

        margin-top: 3px;
    }


    /* =====================================================
       MEDICAL WARNING
       ===================================================== */

    .warning {

        margin-top: 30px;

        background: rgba(255,193,7,0.06);

        border: 1px solid rgba(255,193,7,0.25);

        border-radius: 14px;

        padding: 15px 18px;

        color: #d7dbe0;

        font-size: 13px;

        line-height: 1.6;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {

        text-align: center;

        margin-top: 35px;

        color: #596473;

        font-size: 13px;
    }


    footer {
        visibility: hidden;
    }


    /* =====================================================
       RESPONSIVE
       ===================================================== */

    @media (max-width: 768px) {

        .title {
            font-size: 34px;
        }

        .subtitle {
            font-size: 15px;
        }

        .logo {
            font-size: 42px;
        }

        .result-title {
            font-size: 26px;
        }

        .main .block-container {
            padding: 20px 15px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def loadModel():

    return tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepareImage(uploaded_image):

    img = Image.open(uploaded_image).convert("RGB")

    img_resized = img.resize(IMG_SIZE)

    img_array = np.array(img_resized) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img, img_array


# ============================================================
# PREDICTION
# ============================================================

def predictImage(model, img_array):

    prob = model.predict(img_array, verbose=0)[0][0]

    if prob > 0.5:

        return (
            "PNEUMONIA",
            prob,
            "pneumonia"
        )

    return (
        "NORMAL",
        1 - prob,
        "normal"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="header">

        <div class="logo">🫁</div>

        <h1 class="title">
            PNEUMONIA AI
        </h1>

        <div class="subtitle">
            Chest X-Ray Analysis powered by Artificial Intelligence
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = loadModel()

except Exception as e:

    st.error(
        f"Unable to load the AI model: {e}"
    )

    st.stop()


# ============================================================
# UPLOAD CARD
# ============================================================

st.markdown(
    """
    <div class="card">

        <div class="section-title">
            📤 Upload Chest X-Ray
        </div>

        <div class="section-description">
            Upload a JPG or PNG chest X-ray image for AI analysis.
        </div>

    """,
    unsafe_allow_html=True,
)


uploaded_file = st.file_uploader(
    "Upload X-ray",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)


st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    with st.spinner("AI is analyzing the X-ray..."):

        image, img_array = prepareImage(uploaded_file)

        result, confidence, style = predictImage(
            model,
            img_array
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # ========================================================
    # TWO COLUMN LAYOUT
    # ========================================================

    left, right = st.columns(
        [1.1, 0.9],
        gap="large"
    )


    # ========================================================
    # IMAGE
    # ========================================================

    with left:

        st.markdown(
            """
            <div class="card">

                <div class="section-title">
                    🩻 X-Ray Image
                </div>

                <div class="section-description">
                    Uploaded chest radiograph
                </div>

            """,
            unsafe_allow_html=True,
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown(
            f"""
            <div class="info-box">

                <div class="info-label">
                    Image Status
                </div>

                <div class="info-value">
                    ✓ Successfully processed
                </div>

            </div>

            <div class="info-box">

                <div class="info-label">
                    Model Input
                </div>

                <div class="info-value">
                    224 × 224 pixels
                </div>

            </div>

            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ========================================================
    # RESULT
    # ========================================================

    with right:

        if style == "pneumonia":

            result_icon = "⚠️"

            result_description = (
                "The AI model detected patterns "
                "associated with pneumonia."
            )

            result_class = "pneumonia-card"

            title_class = "pneumonia-title"

        else:

            result_icon = "✓"

            result_description = (
                "The AI model did not detect "
                "patterns associated with pneumonia."
            )

            result_class = "normal-card"

            title_class = "normal-title"


        st.markdown(
            f"""
            <div class="result-card {result_class}">

                <div class="result-icon">
                    {result_icon}
                </div>

                <div class="result-title {title_class}">
                    {result}
                </div>

                <div class="result-description">
                    {result_description}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        st.markdown(
            f"""
            <div class="confidence-label">

                <span>
                    AI CONFIDENCE
                </span>

                <span class="confidence-value">
                    {confidence:.1%}
                </span>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(
            float(confidence)
        )


        # ====================================================
        # MODEL INFO
        # ====================================================

        st.markdown(
            """
            <div class="info-box">

                <div class="info-label">
                    Model
                </div>

                <div class="info-value">
                    Pneumonia Detection CNN
                </div>

            </div>

            <div class="info-box">

                <div class="info-label">
                    Input Format
                </div>

                <div class="info-value">
                    RGB • 224 × 224
                </div>

            </div>

            <div class="info-box">

                <div class="info-label">
                    Classification
                </div>

                <div class="info-value">
                    Normal / Pneumonia
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="warning">

        ⚠️ <b>Medical Disclaimer</b><br>

        This application is an experimental AI demonstration
        and is intended for educational purposes only.
        The prediction should not be considered a medical diagnosis.
        Always consult a qualified healthcare professional
        for proper interpretation of chest X-ray images.

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🫁 Pneumonia AI • Deep Learning Medical Imaging

        <br>

        Experimental Computer Vision Application

    </div>
    """,
    unsafe_allow_html=True,
)
