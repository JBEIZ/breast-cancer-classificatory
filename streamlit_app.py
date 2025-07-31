import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("breast_cancer_classifier_model.keras")

model = load_model()

# Define class names (adjust according to your model)
class_names = ["Benign", "Malignant", "Normal"]

st.title("Breast Cancer Classifier")
st.write("Upload an ultrasound image to classify it as benign, malignant, or normal.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (adjust size as required by your model)
    img_resized = image.resize((224, 224))  # Change (224, 224) to your model input
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_batch)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"### Prediction: {predicted_class}")
    st.bar_chart(prediction[0])




