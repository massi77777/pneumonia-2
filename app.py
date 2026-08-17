"""
تطبيق Streamlit لتشخيص الالتهاب الرئوي من صورة أشعة سينية للصدر
يستخدم النموذج المحفوظ pneumonia_model.keras
نسخة احترافية بواجهة داكنة (Dark Mode)
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime

IMG_SIZE = (224, 224)
MODEL_PATH = "pneumonia_model.keras"


# ============================================================
# ⚙️ إعدادات الصفحة
# ============================================================
st.set_page_config(
    page_title="تشخيص الالتهاب الرئوي | RadioAI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 🎨 التصميم الداكن (Dark Mode)
# ============================================================
def injectCSS():
    st.markdown(
        """
        <style>
        :root{
            --bg-main:#0e1117;
            --bg-card:#161b22;
            --bg-card-hover:#1c2230;
            --border-color:#2a3140;
            --accent:#00c2a8;
            --accent-soft:rgba(0,194,168,0.12);
            --danger:#ff5c6c;
            --danger-soft:rgba(255,92,108,0.12);
            --success:#3ddc97;
            --success-soft:rgba(61,220,151,0.12);
            --text-main:#e6e8ec;
            --text-muted:#8b94a3;
        }

        .stApp{
            background-color: var(--bg-main);
            color: var(--text-main);
        }

        /* الهيدر العلوي */
        header[data-testid="stHeader"]{
            background-color: var(--bg-main);
        }

        /* الشريط الجانبي */
        section[data-testid="stSidebar"]{
            background-color: var(--bg-card);
            border-right: 1px solid var(--border-color);
        }

        /* عنوان رئيسي */
        .hero{
            background: linear-gradient(135deg, rgba(0,194,168,0.10), rgba(0,194,168,0.02));
            border: 1px solid var(--border-color);
            border-radius: 18px;
            padding: 28px 32px;
            margin-bottom: 24px;
        }
        .hero h1{
            font-size: 1.9rem;
            margin-bottom: 6px;
            color: var(--text-main);
        }
        .hero p{
            color: var(--text-muted);
            font-size: 1rem;
            margin: 0;
        }

        /* بطاقات عامة */
        .card{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 22px 24px;
            margin-bottom: 18px;
        }

        /* منطقة رفع الملفات */
        [data-testid="stFileUploaderDropzone"]{
            background-color: var(--bg-card);
            border: 1.5px dashed var(--border-color) !important;
            border-radius: 14px;
        }
        [data-testid="stFileUploaderDropzone"]:hover{
            border-color: var(--accent) !important;
        }

        /* نتيجة التشخيص */
        .result-box{
            border-radius: 16px;
            padding: 22px 26px;
            margin-top: 10px;
            border: 1px solid;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .result-danger{
            background-color: var(--danger-soft);
            border-color: var(--danger);
        }
        .result-success{
            background-color: var(--success-soft);
            border-color: var(--success);
        }
        .result-icon{
            font-size: 2.4rem;
            line-height: 1;
        }
        .result-title{
            font-size: 1.3rem;
            font-weight: 700;
            margin: 0 0 4px 0;
        }
        .result-title.danger{ color: var(--danger); }
        .result-title.success{ color: var(--success); }
        .result-sub{
            color: var(--text-muted);
            font-size: 0.92rem;
            margin: 0;
        }

        /* شارات صغيرة */
        .badge{
            display:inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            background-color: var(--accent-soft);
            color: var(--accent);
            border: 1px solid rgba(0,194,168,0.35);
        }

        /* شريط الثقة */
        .confidence-label{
            display:flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 14px;
            margin-bottom: 4px;
        }

        /* تذييل تحذيري */
        .disclaimer{
            background-color: rgba(255,193,7,0.08);
            border: 1px solid rgba(255,193,7,0.35);
            border-radius: 12px;
            padding: 14px 18px;
            color: #e8c468;
            font-size: 0.88rem;
            margin-top: 24px;
        }

        /* شبكة الخصائص */
        .stat-grid{
            display:grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-top: 8px;
        }
        .stat-item{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 14px 16px;
            text-align:center;
        }
        .stat-item .num{
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--accent);
        }
        .stat-item .lbl{
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-top: 2px;
        }

        /* أزرار Streamlit */
        .stButton>button{
            background-color: var(--accent);
            color: #06231f;
            border: none;
            border-radius: 10px;
            font-weight: 700;
            padding: 0.55rem 1.2rem;
        }
        .stButton>button:hover{
            background-color: #00e0c0;
            color: #06231f;
        }

        /* شريط التقدم */
        .stProgress > div > div{
            background-color: var(--accent) !important;
        }

        footer{ visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 🧠 تحميل النموذج والتنبؤ
# ============================================================
@st.cache_resource
def loadModel():
    """تحميل النموذج مرة واحدة فقط وحفظه في الذاكرة (cache)"""
    return tf.keras.models.load_model(MODEL_PATH)


def prepareImage(uploaded_image):
    """تجهيز الصورة المرفوعة بنفس طريقة تدريب النموذج"""
    img = Image.open(uploaded_image).convert("RGB")
    resized = img.resize(IMG_SIZE)
    img_array = np.array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def predictImage(model, img_array):
    """التنبؤ بالنتيجة ونسبة الثقة"""
    prob = model.predict(img_array)[0][0]
    if prob > 0.5:
        return "PNEUMONIA", "التهاب رئوي", float(prob)
    else:
        return "NORMAL", "طبيعي", float(1 - prob)


# ============================================================
# 🖼️ عناصر الواجهة
# ============================================================
def renderSidebar():
    with st.sidebar:
        st.markdown("### 🩺 RadioAI")
        st.caption("مساعد ذكي لتحليل صور الأشعة")
        st.markdown("---")
        st.markdown("#### 📋 كيف يعمل التطبيق؟")
        st.markdown(
            """
            1. ارفع صورة أشعة سينية للصدر (JPG / PNG)
            2. سيقوم النموذج بتحليل الصورة تلقائيًا
            3. تحصل على النتيجة ونسبة الثقة
            """
        )
        st.markdown("---")
        st.markdown("#### 🧬 عن النموذج")
        st.markdown(
            """
            <span class="badge">MobileNetV2</span>
            <br><br>
            نموذج تعلّم عميق مُدرّب على صور أشعة صدرية
            لتصنيفها بين حالة طبيعية والتهاب رئوي.
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.caption(f"🕓 {datetime.now().strftime('%Y-%m-%d')}")


def renderHero():
    st.markdown(
        """
        <div class="hero">
            <h1>🩺 تشخيص الالتهاب الرئوي من صور الأشعة السينية</h1>
            <p>ارفع صورة أشعة سينية للصدر، وسيقوم النموذج بتحديد ما إذا كانت طبيعية
            أو تدل على وجود التهاب رئوي، مع نسبة الثقة في التنبؤ.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def renderResult(image, label_en, label_ar, confidence):
    col1, col2 = st.columns([1, 1.1], gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, caption="الصورة المرفوعة", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        is_danger = label_en == "PNEUMONIA"
        icon = "⚠️" if is_danger else "✅"
        box_class = "result-danger" if is_danger else "result-success"
        title_class = "danger" if is_danger else "success"

        st.markdown(
            f"""
            <div class="result-box {box_class}">
                <div class="result-icon">{icon}</div>
                <div>
                    <p class="result-title {title_class}">{label_ar} ({label_en})</p>
                    <p class="result-sub">نتيجة التحليل الآلي للصورة المرفوعة</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="confidence-label">
                <span>نسبة الثقة في التنبؤ</span>
                <span><b>{confidence:.1%}</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(confidence, 0.0), 1.0))

        st.markdown(
            f"""
            <div class="stat-grid">
                <div class="stat-item">
                    <div class="num">{IMG_SIZE[0]}×{IMG_SIZE[1]}</div>
                    <div class="lbl">أبعاد المعالجة</div>
                </div>
                <div class="stat-item">
                    <div class="num">MobileNetV2</div>
                    <div class="lbl">بنية النموذج</div>
                </div>
                <div class="stat-item">
                    <div class="num">{confidence:.0%}</div>
                    <div class="lbl">مستوى الثقة</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def renderDisclaimer():
    st.markdown(
        """
        <div class="disclaimer">
            ⚠️ <b>تنويه:</b> هذا التطبيق تجريبي/تعليمي ولا يُغني إطلاقًا عن استشارة
            طبيب مختص أو إجراء الفحوصات الطبية اللازمة. لا تعتمد على نتائجه لاتخاذ
            قرارات علاجية.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ▶️ تشغيل التطبيق
# ============================================================
def main():
    injectCSS()
    renderSidebar()
    renderHero()

    model = loadModel()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📤 رفع صورة الأشعة")
    uploaded_file = st.file_uploader(
        "اسحب الصورة هنا أو اضغط للاختيار (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        with st.spinner("🔎 جاري تحليل الصورة..."):
            image, img_array = prepareImage(uploaded_file)
            label_en, label_ar, confidence = predictImage(model, img_array)
        renderResult(image, label_en, label_ar, confidence)

    renderDisclaimer()


if __name__ == "__main__":
    main()
