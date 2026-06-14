
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
# إعدادات الصفحة
st.set_page_config(
    page_title="Plant Disease Recognition System",
    page_icon="🌿",
    layout="wide"
)
# تحميل ملف العلاجات
with open(r"D:\saved_models\treat.json", "r", encoding="utf-8") as f:
    treatments = json.load(f)
# CSS
css = """
<style>
body {
    background-color: #f5f9f4;
    font-family: 'Segoe UI', sans-serif;
}
.navbar {
    background: white;
    padding: 15px 30px;
    border-radius: 14px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    margin-bottom: 25px;
    display: flex;
    justify-content: space-between;
}
.nav-title {
    font-size: 26px;
    font-weight: 700;
}
.hero {
    background: linear-gradient(135deg, #d9ffd6 0%, #e9faff 100%);
    padding: 80px 20px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 45px;
}
.result-card {
    background: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 6px 25px rgba(0,0,0,0.08);
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)
# Navbar
st.markdown(
    """
    <div class="navbar">
        <div class="nav-title">🌿 Plant Doctor</div>
    </div>
    """,
    unsafe_allow_html=True
)
# Hero
st.markdown(
    """
    <div class="hero">
        <h1>Identify Plant Diseases Instantly</h1>
        <p>ارفع صورة ورقة النبات وسيتم التشخيص مع العلاج</p>
    </div>
    """,
    unsafe_allow_html=True
)
model = load_model("https://drive.google.com/file/d/1fHMVrrJkxC_xlYLOIq-gg6Ca3dX_Rqbf/view?usp=drive_link")
CLASS_NAMES = [
   "Apple___Apple_scab",
   "Apple___Black_rot", 
   "Apple___Cedar_apple_rust",
   "Apple___healthy",
   "Blueberry___healthy",
   "Cherry_(including_sour)___Powdery_mildew",
   "Cherry_(including_sour)___healthy",
   "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
   "Corn_(maize)___Common_rust_",
   "Corn_(maize)___Northern_Leaf_Blight",
   "Corn_(maize)___healthy",
   "Grape___Black_rot",
   "Grape___Esca_(Black_Measles)",
   "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 
   "Grape___healthy",
   "Orange___Haunglongbing_(Citrus_greening)",
   "Peach___Bacterial_spot",
   "Peach___healthy",
   "Pepper,_bell___Bacterial_spot",
   "Pepper,_bell___healthy",
   "Potato___Early_blight",
   "Potato___Late_blight", 
   "Potato___healthy",
   "Raspberry___healthy",
   "Soybean___healthy",
   "Squash___Powdery_mildew",
   "Strawberry___Leaf_scorch",
   "Strawberry___healthy",
   "Tomato___Bacterial_spot",
   "Tomato___Early_blight",
   "Tomato___Late_blight",
   "Tomato___Leaf_Mold",
   "Tomato___Septoria_leaf_spot",
   "Tomato___Spider_mites Two-spotted_spider_mite",
   "Tomato___Target_Spot",
   "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
   "Tomato___Tomato_mosaic_virus", 
   "Tomato___healthy"
]
# prediction
def predict(img):
    img = img.resize((256, 256))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    pred = model.predict(arr)

    idx = np.argmax(pred)
    conf = np.max(pred)

    raw_name = CLASS_NAMES[idx]
    name = raw_name.replace("___", " - ").replace("_", " ").replace("healthy", "سليم")

    return name, conf
# UI
left, right = st.columns(2)

with left:
    st.header("📸 Upload Image")
    upl = st.file_uploader("Choose a leaf image", type=["jpg","jpeg","png"])

    if upl:
        img = Image.open(upl)
        st.image(img, caption="Uploaded Image", use_column_width=True)

with right:
    st.header("🌱 Diagnosis")

    if upl:
        name, conf = predict(img)

        # جلب العلاج
        data = treatments.get(name, None)

        desc = "لا يوجد وصف"
        treat = "لا يوجد علاج"
        prev = "لا توجد نصائح"

        if data:
            desc = data["description"]
            treat = data["treatment"]
            prev = data["prevention"]

        st.markdown(
            f"""
            <div class="result-card">
                <h3 style="color:#1b8d62;">{name}</h3>
                <hr>
                <b>Confidence:</b> {conf*100:.2f}%
                <hr>
                <b>Description:</b>
                <p>{desc}</p>
                <b>Treatment:</b>
                <p>{treat}</p>
                <b>Prevention:</b>
                <p>{prev}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.warning("⚠️ النتائج استرشادية وليست بديلاً عن استشارة مهندس زراعي")

    else:
        st.info("Upload an image to see the prediction.")
# Footer
st.markdown(
    """
    <div style="text-align:center; margin-top:50px; color:#777;">
        Powered by Deep Learning 🌿
    </div>
    """,
    unsafe_allow_html=True
)
