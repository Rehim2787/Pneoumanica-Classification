import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("C:/Users/Asus/OneDrive/Documents/Data Science/DeepLearning_project/last_model.h5")

# Streamlit app layout
st.title('Image Classification App')

uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_img is not None:
    img = Image.open(uploaded_img)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize the image to match the model's input shape (256x256)
    img = img.resize((256, 256))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    classes = model.predict(img_data)
    predicted_label = 1 if classes[0][0] >= 0.5 else 0

    if predicted_label == 0:
        prediction = "Normal"
    else:
        prediction = "Pneumonia"

    confidence = (abs((classes[0][0]-50)/50))*100 # Use the probability as confidence

    st.write("Prediction: ", prediction)
    st.write("Confidence: {:.2f}%".format(confidence))
