import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# ============================================================================
# PAGE CONFIGURATION (حافظت على إعداداتك)
# ============================================================================
st.set_page_config(
    page_title="Age, Gender & Ethnicity Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS (حافظت على التنسيق الجميل اللي عملته)
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS & CACHED MODEL LOAD
# ============================================================================
IMG_SIZE = 64
GENDER_MAP = {0: 'Male 👨', 1: 'Female 👩'}
ETHNICITY_MAP = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}

@st.cache_resource
def load_model_file():
    """تحميل الموديل مرة واحدة لضمان استقرار السيرفر"""
    try:
        # غيرت الاسم لـ best_vgg16.keras عشان يشتغل مع ملفك
        model = tf.keras.models.load_model('best_vgg16.keras')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ============================================================================
# PREPROCESSING & PREDICTION
# ============================================================================
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def make_prediction(model, image_tensor):
    predictions = model.predict(image_tensor, verbose=0)
    
    # استخراج النتائج (بناءً على تصميم الـ Multi-head الخاص بك)
    age_pred = int(np.round(predictions[0][0][0]))
    gender_raw = predictions[1][0][0]
    gender_pred = 1 if gender_raw > 0.5 else 0
    gender_conf = gender_raw if gender_pred == 1 else (1 - gender_raw)
    
    ethnicity_probs = predictions[2][0]
    eth_pred = np.argmax(ethnicity_probs)
    eth_conf = ethnicity_probs[eth_pred]
    
    return {
        'age': age_pred,
        'gender': gender_pred,
        'gender_confidence': float(gender_conf),
        'ethnicity': eth_pred,
        'ethnicity_confidence': float(eth_conf),
        'ethnicity_probs': ethnicity_probs
    }

# ============================================================================
# MAIN APP (التصميم الأصلي بتاعك)
# ============================================================================
def main():
    st.markdown("<h1 class='main-header'>🧠 Age, Gender & Ethnicity Predictor</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 About")
        st.info("This app uses a **Multi-Head CNN** to predict Age, Gender, and Ethnicity.")
        st.markdown("## ⚙️ Settings")
        show_confidence = st.checkbox("Show confidence scores", value=True)

    model = load_model_file()
    if model is None: st.stop()

    uploaded_file = st.file_uploader("Choose a face image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        processed_img = preprocess_image(image)
        results = make_prediction(model, processed_img)

        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Uploaded Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Predictions")
            
            # Age
            st.markdown(f"<div class='prediction-box'>👴 <b>Age:</b> <span class='metric-value'>{results['age']} years</span></div>", unsafe_allow_html=True)
            
            # Gender
            st.markdown(f"<div class='prediction-box'>👥 <b>Gender:</b> <span class='metric-value'>{GENDER_MAP[results['gender']]}</span></div>", unsafe_allow_html=True)
            if show_confidence: st.progress(results['gender_confidence'])
            
            # Ethnicity
            st.markdown(f"<div class='prediction-box'>🌍 <b>Ethnicity:</b> <span class='metric-value'>{ETHNICITY_MAP[results['ethnicity']]}</span></div>", unsafe_allow_html=True)
            if show_confidence: st.progress(results['ethnicity_confidence'])

        # Download Button (اللمسة بتاعتك)
        st.markdown("---")
        download_text = f"Age: {results['age']}\nGender: {GENDER_MAP[results['gender']]}\nEthnicity: {ETHNICITY_MAP[results['ethnicity']]}"
        st.download_button("📥 Download Results", download_text, file_name="results.txt")

if __name__ == "__main__":
    main()