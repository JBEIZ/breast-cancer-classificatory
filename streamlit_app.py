import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('breast_cancer_classifier_model.keras')

# Class names mapping
class_names = ['normal', 'benign', 'malignant']

# Function to process the image and make predictions
def predict_image(image):
    # Resize and normalize the image
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(image)

    # Get the class with the highest probability
    predicted_class = np.argmax(prediction, axis=-1)  # Get class index with max probability
    predicted_label = class_names[predicted_class[0]]  # Get label for predicted class
    
    return predicted_label, prediction[0]  # Return label and class probabilities

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
