import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("breast_cancer_classifier_model.keras")
    return model

model = load_model()

# Define class names (update if needed)
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']

# Preprocess image
def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict function
def predict(img):
    processed = preprocess_image(img)
    preds = model.predict(processed)
    predicted_class = np.argmax(preds, axis=-1)[0]
    confidence = float(np.max(preds))
    return CLASS_NAMES[predicted_class], confidence

# Streamlit UI
st.title("🩺 Breast Cancer Classifier")
st.write("Upload a breast ultrasound image to classify as **Benign**, **Malignant**, or **Normal**.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        label, confidence = predict(image)
        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence * 100:.2f}%")



