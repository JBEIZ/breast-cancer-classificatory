import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import os

# Google Drive file ID (replace with your actual file ID)
file_id = '1DV8LS_uSzQ_3hRrcTEmu7FnXkaMSzfOg'
model_path = 'breast_cancer_classifier_model.keras'

# Check if the model is already downloaded
if not os.path.exists(model_path):
    with st.spinner("Downloading the model from Google Drive..."):
        gdown.download(f'https://drive.google.com/uc?id={file_id}', model_path, quiet=False)

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Class names
class_names = ['normal', 'benign', 'malignant']

# Prediction function
def predict_image(image):
    try:
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        if image.shape[-1] == 4:  # Handle images with alpha channel
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=-1)
        predicted_label = class_names[predicted_class[0]]
        return predicted_label, prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Streamlit interface
st.title('🧠 Breast Cancer Classification (Ultrasound Image)')
st.write('Upload an ultrasound image to classify it as **normal**, **benign**, or **malignant**.')

uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('🧪 Predict'):
        with st.spinner("Analyzing image..."):
            predicted_label, prediction_probabilities = predict_image(image)
        
        if predicted_label:
            st.success(f'✅ **Prediction:** {predicted_label.capitalize()}')
            st.subheader('📊 Class Probabilities:')
            for i, class_name in enumerate(class_names):
                st.write(f"- **{class_name.capitalize()}**: {prediction_probabilities[i]:.4f}")


