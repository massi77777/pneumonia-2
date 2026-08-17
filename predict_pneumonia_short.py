"""
نسخة مختصرة من مشروع تشخيص الالتهاب الرئوي
تم استبدال الأجزاء الطويلة (تحميل الصور يدويًا، التطبيع، الفحص...) بدوال جاهزة من TensorFlow/Keras
لتقليل عدد الأسطر مع الحفاظ على نفس الفكرة والنتيجة.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

DATA_PATH = "chest_xray_data/chest_xray"
IMG_SIZE = (224, 224)
BATCH = 32


def loadData():
    """تحميل الصور وفك الضغط بأمرين جاهزين بدل عدة أسطر"""
    get_ipython().system('wget -q "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded" -O chest_xray.zip')
    get_ipython().system('unzip -o -q chest_xray.zip -d chest_xray_data')  # -o = استبدال تلقائي بدون سؤال (يمنع مشكلة "replace ...?")


def prepareData():
    """تجهيز بيانات train/val/test بدالة جاهزة واحدة بدل ImageDataGenerator كاملة"""
    train_data = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_PATH}/train", image_size=IMG_SIZE, batch_size=BATCH,
        label_mode="binary", validation_split=0.15, subset="training", seed=1
    )
    val_data = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_PATH}/train", image_size=IMG_SIZE, batch_size=BATCH,
        label_mode="binary", validation_split=0.15, subset="validation", seed=1
    )
    test_data = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_PATH}/test", image_size=IMG_SIZE, batch_size=BATCH,
        label_mode="binary", shuffle=False
    )
    # تطبيع القيم (0-255 → 0-1) بدالة جاهزة بدل rescale يدوي
    normalize = layers.Rescaling(1.0 / 255)
    train_data = train_data.map(lambda x, y: (normalize(x), y))
    val_data = val_data.map(lambda x, y: (normalize(x), y))
    test_data = test_data.map(lambda x, y: (normalize(x), y))
    return train_data, val_data, test_data


def buildModel():
    """بناء النموذج: MobileNetV2 جاهز + طبقة تصنيف بسيطة"""
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model, base_model


def trainModel(model, train_data, val_data, epochs=10):
    """تدريب النموذج مع موازنة الفئات تلقائيًا"""
    labels = np.concatenate([y.numpy() for _, y in train_data])
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels.flatten())
    class_weights = dict(enumerate(weights))
    model.fit(train_data, validation_data=val_data, epochs=epochs, class_weight=class_weights)
    return class_weights


def evaluateModel(model, test_data):
    """تقييم جاهز بسطر واحد (model.evaluate) بدل حساب التقرير يدويًا"""
    loss, accuracy = model.evaluate(test_data)
    print(f"Accuracy: {accuracy:.2%}")


def fineTuneModel(model, base_model, train_data, val_data, class_weights):
    """فتح آخر 30 طبقة وإعادة التدريب بمعدل تعلّم صغير"""
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_data, validation_data=val_data, epochs=5, class_weight=class_weights)


def predictImage(model, image_path):
    """التنبؤ بصورة واحدة باستخدام دوال Keras الجاهزة لتحميل الصور"""
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prob = model.predict(img_array)[0][0]
    result = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    confidence = prob if prob > 0.5 else 1 - prob
    print(f"النتيجة: {result} | الثقة: {confidence:.2%}")


def saveModel(model, path="pneumonia_model.keras"):
    """حفظ النموذج بسطر واحد"""
    model.save(path)
    print("تم الحفظ:", path)


# ============================================================
# ▶️ تشغيل المشروع كاملاً بعدد أسطر قليل جدًا
# ============================================================
if __name__ == "__main__":
    loadData()
    train_data, val_data, test_data = prepareData()
    model, base_model = buildModel()
    class_weights = trainModel(model, train_data, val_data)
    evaluateModel(model, test_data)
    fineTuneModel(model, base_model, train_data, val_data, class_weights)
    evaluateModel(model, test_data)
    saveModel(model)
