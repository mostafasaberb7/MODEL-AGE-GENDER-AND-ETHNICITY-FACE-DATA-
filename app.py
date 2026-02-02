"""
🧠 Age, Gender & Ethnicity Prediction App
==========================================
Streamlit deployment for multi-head CNN model
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Age, Gender & Ethnicity Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
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
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 5px;
        height: 20px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

IMG_SIZE = 64
GENDER_MAP = {0: 'Male 👨', 1: 'Female 👩'}
ETHNICITY_MAP = {
    0: 'White',
    1: 'Black',
    2: 'Asian',
    3: 'Indian',
    4: 'Others'
}

# ============================================================================
# LOAD MODEL (with caching)
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    try:
        model = tf.keras.models.load_model('best_multihead_model.keras')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("💡 Make sure 'best_multihead_model.keras' is in the same directory!")
        return None

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image):
    """
    Preprocess uploaded image for model prediction.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        Preprocessed image tensor
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize to model input size
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

# ============================================================================
# PREDICTION
# ============================================================================

def make_prediction(model, image):
    """
    Make prediction on preprocessed image.
    
    Args:
        model: Loaded Keras model
        image: Preprocessed image tensor
        
    Returns:
        Dictionary with predictions
    """
    # Get predictions
    predictions = model.predict(image, verbose=0)
    
    # Extract results
    age_pred = predictions[0][0][0]
    gender_pred = predictions[1][0][0]
    ethnicity_probs = predictions[2][0]
    
    # Process predictions
    predicted_age = int(np.round(age_pred))
    predicted_gender = 1 if gender_pred > 0.5 else 0
    gender_confidence = gender_pred if predicted_gender == 1 else (1 - gender_pred)
    predicted_ethnicity = np.argmax(ethnicity_probs)
    ethnicity_confidence = ethnicity_probs[predicted_ethnicity]
    
    return {
        'age': predicted_age,
        'gender': predicted_gender,
        'gender_confidence': float(gender_confidence),
        'ethnicity': predicted_ethnicity,
        'ethnicity_confidence': float(ethnicity_confidence),
        'ethnicity_probs': ethnicity_probs
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def display_results(image, predictions):
    """Display prediction results with nice formatting"""
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Uploaded Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Predictions")
        
        # Age prediction
        st.markdown("#### 👴 Age")
        st.markdown(f"<div class='prediction-box'>"
                   f"<span class='metric-value'>{predictions['age']} years old</span>"
                   f"</div>", unsafe_allow_html=True)
        
        # Gender prediction
        st.markdown("#### 👥 Gender")
        gender_label = GENDER_MAP[predictions['gender']]
        gender_conf = predictions['gender_confidence'] * 100
        st.markdown(f"<div class='prediction-box'>"
                   f"<span class='metric-value'>{gender_label}</span><br>"
                   f"Confidence: {gender_conf:.1f}%"
                   f"</div>", unsafe_allow_html=True)
        st.progress(predictions['gender_confidence'])
        
        # Ethnicity prediction
        st.markdown("#### 🌍 Ethnicity")
        ethnicity_label = ETHNICITY_MAP[predictions['ethnicity']]
        ethnicity_conf = predictions['ethnicity_confidence'] * 100
        st.markdown(f"<div class='prediction-box'>"
                   f"<span class='metric-value'>{ethnicity_label}</span><br>"
                   f"Confidence: {ethnicity_conf:.1f}%"
                   f"</div>", unsafe_allow_html=True)
        st.progress(predictions['ethnicity_confidence'])
        
        # Ethnicity breakdown
        with st.expander("📊 See all ethnicity probabilities"):
            for i, prob in enumerate(predictions['ethnicity_probs']):
                st.write(f"{ETHNICITY_MAP[i]}: {prob*100:.2f}%")
                st.progress(float(prob))

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("<h1 class='main-header'>🧠 Age, Gender & Ethnicity Predictor</h1>", 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        Upload a face image and let AI predict the age, gender, and ethnicity!
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 About")
        st.info("""
        This app uses a **Multi-Head CNN** trained on the UTK Face dataset to predict:
        - 👴 **Age** (1-116 years)
        - 👥 **Gender** (Male/Female)
        - 🌍 **Ethnicity** (5 categories)
        
        **Model Architecture:**
        - 3 Convolutional blocks
        - 3 task-specific heads
        - ~1M parameters
        """)
        
        st.markdown("## ⚙️ Settings")
        show_confidence = st.checkbox("Show confidence scores", value=True)
        
        st.markdown("## 💡 Tips")
        st.markdown("""
        - Use **clear face images**
        - **Good lighting** improves accuracy
        - **Front-facing** photos work best
        """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a face image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear face image for best results"
    )
    
    # Demo images option
    st.markdown("### 🖼️ Or try a demo image:")
    demo_option = st.selectbox(
        "Select demo image",
        ["None", "Demo 1", "Demo 2", "Demo 3"]
    )
    
    # Process image
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        # Preprocess
        with st.spinner("🔄 Processing image..."):
            processed_image = preprocess_image(image)
        
        # Predict
        with st.spinner("🧠 Making predictions..."):
            predictions = make_prediction(model, processed_image)
        
        # Display results
        st.markdown("---")
        display_results(image, predictions)
        
        # Download button for results
        st.markdown("---")
        results_text = f"""
Age, Gender & Ethnicity Prediction Results
==========================================

Age: {predictions['age']} years
Gender: {GENDER_MAP[predictions['gender']]} (Confidence: {predictions['gender_confidence']*100:.1f}%)
Ethnicity: {ETHNICITY_MAP[predictions['ethnicity']]} (Confidence: {predictions['ethnicity_confidence']*100:.1f}%)

All Ethnicity Probabilities:
{chr(10).join([f"- {ETHNICITY_MAP[i]}: {prob*100:.2f}%" for i, prob in enumerate(predictions['ethnicity_probs'])])}
        """
        
        st.download_button(
            label="📥 Download Results",
            data=results_text,
            file_name="prediction_results.txt",
            mime="text/plain"
        )
    
    elif demo_option != "None":
        st.info("Demo images not available. Please upload your own image.")
    
    else:
        # Instructions
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background-color: #f0f2f6; border-radius: 10px; margin: 2rem 0;'>
            <h3>👆 Upload an image to get started!</h3>
            <p>The AI will analyze the face and predict age, gender, and ethnicity.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        Made with ❤️ using TensorFlow & Streamlit | 
        <a href='https://github.com' target='_blank'>View on GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()