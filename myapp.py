import streamlit as st
import os


st.title("Deteksi Chest X-ray 📸")
st.header("Identifikasi apa yang ada di gambar berikut!")

# Pick the model version
choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1 (CNN)",
     "Model 2 (VGG-16)", 
     "Model 3 (Xception)")
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
    st.image(uploaded_file, use_column_width=True)
    pred_button = st.button("Prediksi")