import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load your saved model
model = load_model('C:\Users\Joshua\Downloads\breast_cancer_classifier_model.keras')  # or model_name.h5 if you saved in HDF5 format

# Function to process the image and make predictions
def predict_image(image):
    image = image.resize((224, 224))  # Resize to the input shape of the model
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return prediction

# Streamlit interface
st.title('Breast Cancer Classification')
st.write('Upload an ultrasound image to classify it as normal, benign, or malignant.')

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        prediction = predict_image(image)
        st.write('Prediction:', prediction)
