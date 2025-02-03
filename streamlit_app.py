import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os

# Define model path
model_path = "breast_cancer_classifier_model.keras"

model_url = "https://drive.google.com/file/d/1gnQf61mbXKdVrjIKETbbE7tcUUbn8S-O/view?usp=drive_link"
response = requests.get(model_url)
with open(model_path, "wb") as file:
     file.write(response.content)

# Load the trained model
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    st.error("Model file not found. Please upload it to the same directory.")

# Class names mapping
class_names = ['normal', 'benign', 'malignant']

# Function to process the image and make predictions
def predict_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=-1)[0]
    predicted_label = class_names[predicted_class]
    return predicted_label, prediction[0]

# Streamlit interface
st.title('Breast Cancer Classification')
st.write('Upload an ultrasound image to classify it as normal, benign, or malignant.')

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        predicted_label, prediction_probabilities = predict_image(image)
        st.write(f'Prediction: {predicted_label}')
        
        # Display probabilities for each class
        st.write('Probabilities:')
        for i, class_name in enumerate(class_names):
            st.write(f'{class_name}: {prediction_probabilities[i]:.4f}')
