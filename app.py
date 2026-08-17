"""
تطبيق Streamlit الاحترافي لتشخيص الالتهاب الرئوي
يستخدم النموذج المحفوظ pneumonia_model.keras
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# ⚙️ إعدادات الصفحة (يجب أن تكون في السطر الأول دائمًا)
# ============================================================
st.set_page_config(
    page_title="Pneumo-Scan | تشخيص الالتهاب الرئوي",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 🎨 تنسيق Dark Mode الاحترافي (Custom CSS)
# ============================================================
st.markdown("""
<style>
    /* تخصيص الخلفية الرئيسية لتكون داكنة بشكل احترافي */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* تحسين شكل صندوق رفع الملفات */
    .stFileUploader > div > div > div > div {
        background-color: #1A1C23;
        border: 2px dashed #4CAF50;
        border-radius: 12px;
        padding: 20px;
    }
    
    /* تكبير وتلوين الأرقام في بطاقات النتائج (Metrics) */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #4CAF50;
    }
    
    /* تدوير زوايا التنبيهات */
    .stAlert {
        border-radius: 10px;
    }
    
    /* تحسين خطوط العناوين */
    h1, h2, h3 {
        color: #E0E0E0 !important;
    }
    
    /* خط فاصل أنيق */
    hr {
        border-color: #2D303E;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 🧠 إعدادات النموذج والوظائف الأساسية
# ============================================================
IMG_SIZE = (224, 224)
MODEL_PATH = "pneumonia_model.keras"

@st.cache_resource(show_spinner=False)
def load_model():
    """تحميل النموذج مرة واحدة مع معالجة الأخطاء"""
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"⚠️ تعذر تحميل النموذج من المسار '{MODEL_PATH}'. الرجاء التأكد من وجود الملف بجانب الكود.")
        st.stop()

def prepare_image(uploaded_image):
    """تجهيز الصورة المرفوعة لتطابق مدخلات نموذج الذكاء الاصطناعي"""
    img = Image.open(uploaded_image).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def predict_image(model, img_array):
    """استنتاج النتيجة ونسبة الثقة"""
    prob = model.predict(img_array)[0][0]
    if prob > 0.5:
        return "PNEUMONIA", prob
    else:
        return "NORMAL", 1 - prob

# ============================================================
# 🖥️ واجهة المستخدم
# ============================================================
def main():
    # --- الشريط الجانبي (Sidebar) ---
    with st.sidebar:
        st.title("🩺 معلومات النظام")
        st.info(
            "يعتمد هذا النظام على شبكات عصبية التفافية (CNN) تم تدريبها "
            "لتحليل صور الأشعة السينية للصدر (Chest X-Ray) واكتشاف علامات الالتهاب الرئوي بدقة عالية."
        )
        st.markdown("---")
        st.warning(
            "⚠️ **إخلاء مسؤولية طبية:**\n\n"
            "هذا التطبيق مصمم للأغراض البحثية والتجريبية فقط. "
            "النتائج الصادرة لا تُغني إطلاقًا عن استشارة طبيب أو أخصائي أمراض صدرية."
        )
        st.markdown("---")
        st.caption("تم التطوير بواسطة Streamlit & TensorFlow")

    # --- المنطقة الرئيسية (Main Area) ---
    st.title("🫁 النظام الذكي للتحليل الإشعاعي (Pneumo-Scan)")
    st.write("يرجى رفع صورة أشعة سينية (X-Ray) للصدر ليقوم محرك الذكاء الاصطناعي بتحليلها فوراً.")

    # تحميل النموذج
    with st.spinner("جاري تهيئة محرك الذكاء الاصطناعي..."):
        model = load_model()

    # أداة رفع الملفات
    uploaded_file = st.file_uploader("قم بإفلات أو اختيار صورة الأشعة هنا (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.markdown("---")
        st.subheader("📊 التقرير التشخيصي للذكاء الاصطناعي")

        # استخدام الأعمدة لترتيب الصورة بجانب النتائج بشكل احترافي
        col1, col2 = st.columns([1, 1.2])

        with st.spinner("جاري فحص الأنسجة وتحليل الصورة..."):
            image, img_array = prepare_image(uploaded_file)
            result, confidence = predict_image(model, img_array)

        # العمود الأول: عرض الصورة
        with col1:
            st.image(image, caption="صورة الأشعة المرفوعة", use_container_width=True)

        # العمود الثاني: عرض النتائج
        with col2:
            st.markdown("#### النتيجة النهائية:")
            
            if result == "PNEUMONIA":
                st.error("🚨 **تنبيه:** يشير التحليل إلى وجود **التهاب رئوي (Pneumonia)**.")
                st.progress(float(confidence), text="مؤشر الاحتمالية")
            else:
                st.success("✅ **سليم:** الرئتان تبدوان **بحالة طبيعية (Normal)** ولا توجد علامات التهاب.")
                st.progress(float(confidence), text="مؤشر الاحتمالية")

            st.markdown("<br>", unsafe_allow_html=True) # مسافة فارغة
            
            # عرض نسبة الثقة بشكل رياضي بارز
            st.metric(label="مستوى ثقة النموذج (AI Confidence)", value=f"{confidence:.2%}")

if __name__ == "__main__":
    main()
