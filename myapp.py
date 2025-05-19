import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import keras

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


st.title("Deteksi Chest X-ray 📸")
st.header("Identifikasi apa yang ada di gambar berikut!")

# Pick the model version
choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1 (VGG-19)",
     "Model 2 (Xception)", 
     "Model 3 (VGG-16)",
     "Model 4 (CNN)")
     )

# Display info about model and classes
if st.checkbox("Show classes"):
    st.write(f"You chose model, these are the classes of images it can identify:\n")
    # st.write(f"You chose {MODEL}, these are the classes of food it can identify:\n", CLASSES)

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload an image of chest x-ray",
                                 type=["png", "jpeg", "jpg"])

# Create logic for app flow
if not uploaded_file:
    st.warning("Silahkan Masukkan Gambar.")
    st.stop()
else:
    # session_state.uploaded_image = uploaded_file.read()
    # st.image(session_state.uploaded_image, use_column_width=True)
    st.session_state.uploaded_image = uploaded_file.read()
    st.image(st.session_state.uploaded_image, use_container_width=True)
    pred_button = st.button("Prediksi")
    # st.image(uploaded_file, use_container_width=True)
    # pred_button = st.button("Prediksi")

if pred_button:
    # Load the selected model
    if choose_model == "Model 1 (VGG-19)":
        model = tf.keras.models.load_model('tb-vgg19.keras')
    elif choose_model == "Model 2 (Xception)":
        model = tf.keras.models.load_model('tb-exception.keras')
    elif choose_model == "Model 3 (VGG-16)":
        model = tf.keras.models.load_model('tb-vgg16.keras')
    elif choose_model == "Model 4 (CNN)":
        model = tf.keras.models.load_model('tb-cnn.keras')


        # Preprocess the image
        image = Image.open(uploaded_file)
        image = image.resize((224, 224))  # Resize image to match model input size
        image_array = np.array(image) / 255.0  # Normalize image
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make predictions
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)  # multi-class

        # Mapping class
        class_labels = ['Normal', 'Laten', 'Aktif']  

        predicted_label = class_labels[predicted_class[0]]

        # Display the result
        st.success(f"Hasil Prediksi: {predicted_label}")
        st.info(f"Confidence: {np.max(prediction[0])}")
 
        st.balloons()