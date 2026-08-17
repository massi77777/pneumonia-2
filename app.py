"""
تطبيق Streamlit لتشخيص الالتهاب الرئوي من صورة أشعة سينية للصدر
يستخدم النموذج المحفوظ pneumonia_model.keras
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = "pneumonia_model.keras"


@st.cache_resource
def loadModel():
    """تحميل النموذج مرة واحدة فقط وحفظه في الذاكرة (cache)"""
    return tf.keras.models.load_model(MODEL_PATH)


def prepareImage(uploaded_image):
    """تجهيز الصورة المرفوعة بنفس طريقة تدريب النموذج"""
    img = Image.open(uploaded_image).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def predictImage(model, img_array):
    """التنبؤ بالنتيجة ونسبة الثقة"""
    prob = model.predict(img_array)[0][0]
    if prob > 0.5:
        return "PNEUMONIA (التهاب رئوي)", prob
    else:
        return "NORMAL (طبيعي)", 1 - prob


def showResult(image, result, confidence):
    """عرض الصورة والنتيجة في الواجهة"""
    st.image(image, caption="الصورة المرفوعة", use_container_width=True)
    if "PNEUMONIA" in result:
        st.error(f"النتيجة: {result}")
    else:
        st.success(f"النتيجة: {result}")
    st.write(f"نسبة الثقة: **{confidence:.2%}**")


# ============================================================
# ▶️ واجهة التطبيق
# ============================================================
st.set_page_config(page_title="تشخيص الالتهاب الرئوي", page_icon="🩺")

st.title("🩺 تشخيص الالتهاب الرئوي من صور الأشعة السينية")
st.write("ارفع صورة أشعة سينية للصدر وسيقوم النموذج بتحديد إذا كانت طبيعية أو تدل على التهاب رئوي.")

model = loadModel()

uploaded_file = st.file_uploader("اختر صورة (jpg / png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("جاري تحليل الصورة..."):
        image, img_array = prepareImage(uploaded_file)
        result, confidence = predictImage(model, img_array)
    showResult(image, result, confidence)

st.markdown("---")
st.caption("⚠️ هذا التطبيق تجريبي ولا يُغني إطلاقًا عن استشارة طبيب مختص.")
