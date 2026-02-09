import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# ============================================================================
# 1. PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Age, Gender & Ethnicity Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 2. CONSTANTS & MAPS
# ============================================================================
IMG_SIZE = 64
GENDER_MAP = {0: 'Male 👨', 1: 'Female 👩'}
ETHNICITY_MAP = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}

# ============================================================================
# 3. LOAD MODEL (Cached to save RAM)
# ============================================================================
@st.cache_resource
def load_trained_model():
    """Load the model once and keep it in memory"""
    try:
        # تأكدنا أن الاسم مطابق للملف المرفوع في الـ GitHub
        model = tf.keras.models.load_model('best_vgg16.keras')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ============================================================================
# 4. IMAGE PREPROCESSING
# ============================================================================
def preprocess_image(image):
    # تحويل الصورة إلى RGB إذا كانت بصيغة مختلفة
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # تحويل إلى Array وتصغير الحجم
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # التطبيع (Normalization) إضافة بُعد الـ Batch
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ============================================================================
# 5. CSS STYLING
# ============================================================================
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; }
    .prediction-box { background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin-bottom: 10px; }
    .metric-value { font-size: 1.5rem; font-weight: bold; color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 6. MAIN APP LOGIC
# ============================================================================
def main():
    st.markdown("<h1 class='main-header'>🧠 Age, Gender & Ethnicity Predictor</h1>", unsafe_allow_html=True)
    
    # Sidebar Info
    st.sidebar.info("This app uses a Multi-Head CNN (VGG16-based) to predict Age, Gender, and Ethnicity.")
    
    model = load_trained_model()
    if model is None:
        st.warning("Model file 'best_vgg16.keras' not found. Please check your GitHub repository.")
        st.stop()

    uploaded_file = st.file_uploader("Upload a face image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display Columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            with st.spinner("Analyzing..."):
                processed_img = preprocess_image(image)
                predictions = model.predict(processed_img, verbose=0)
                
                # توزيح النتائج بناءً على تصميم الـ Multi-head
                # تأكد من ترتيب المخرجات (Outputs) في موديلك
                age_p = int(np.round(predictions[0][0][0]))
                gender_p = 1 if predictions[1][0][0] > 0.5 else 0
                ethnicity_p = np.argmax(predictions[2][0])

                # عرض النتائج
                st.markdown("### 🎯 Prediction Results")
                
                st.markdown(f"<div class='prediction-box'>👴 <b>Age:</b> <span class='metric-value'>{age_p}</span> years</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='prediction-box'>👥 <b>Gender:</b> <span class='metric-value'>{GENDER_MAP[gender_p]}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='prediction-box'>🌍 <b>Ethnicity:</b> <span class='metric-value'>{ETHNICITY_MAP[ethnicity_p]}</span></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()