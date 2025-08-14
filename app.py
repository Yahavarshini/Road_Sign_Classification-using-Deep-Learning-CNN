import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
#import cv2

# Load model and labels
model = load_model("road_sign_model.h5")
sign_names = pd.read_csv("Road_signs/signname.csv")
label_dict = dict(zip(sign_names['ClassId'], sign_names['SignName']))

st.set_page_config(page_title="Road Sign Classifier", layout="centered")
st.title("Road Sign Classifier")
st.markdown("Upload an image of a traffic sign, and the model will predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 32, 32, 3)

    # Make prediction
    pred_probs = model.predict(img_array)
    pred_class = np.argmax(pred_probs)
   

    st.success(f"**Prediction:** {label_dict[pred_class]}")
  
