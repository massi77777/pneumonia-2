# 🩺 تشخيص الالتهاب الرئوي (Pneumonia Detection)

تطبيق Streamlit يستعمل نموذج Deep Learning (MobileNetV2) للتنبؤ إن كانت صورة أشعة الصدر طبيعية أو تدل على التهاب رئوي.

## 📁 محتوى المشروع
- `app.py` — تطبيق Streamlit (الواجهة والتنبؤ)
- `pneumonia_model.keras` — النموذج المدرَّب (يجب إضافته أنت من نتيجة التدريب)
- `requirements.txt` — المكتبات المطلوبة
- `predict_pneumonia_short.py` — كود تدريب النموذج (Colab)

## 🚀 التشغيل محليًا
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ النشر على Streamlit Cloud عبر GitHub
1. أنشئ Repository جديد على GitHub وارفع هذه الملفات (بما فيها `pneumonia_model.keras`).
2. اذهب إلى https://share.streamlit.io وسجّل الدخول بحساب GitHub.
3. اضغط **New app**، اختر الـ Repository، وحدد الملف الرئيسي `app.py`.
4. اضغط **Deploy** وانتظر بضع دقائق.

## ⚠️ ملاحظة
هذا التطبيق تجريبي/تعليمي ولا يُستخدم كتشخيص طبي حقيقي.
