import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Load model YOLOv5
@st.cache_resource
def load_model():
    model_path = "yolov5_best_model.pt"  # Sesuaikan dengan nama model yang ada di repo
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
    return model

model = load_model()

# Fungsi untuk deteksi kebakaran
def detect_fire(image):
    results = model(image)
    return results

# Streamlit UI
st.title("🔥 Deteksi Kebakaran Hutan (WILDFIRE) 🔥")

# Opsi upload gambar atau menggunakan kamera
option = st.radio("Pilih Input:", ["Gunakan Kamera", "Upload Gambar"])

if option == "Gunakan Kamera":
    camera_image = st.camera_input("Ambil Gambar")
    if camera_image:
        image = Image.open(camera_image)
        image = np.array(image)  # Konversi ke format numpy
        results = detect_fire(image)

        # Konversi hasil ke gambar dengan bounding box
        st.image(results.render()[0], caption="Hasil Deteksi Kebakaran", use_column_width=True)

elif option == "Upload Gambar":
    uploaded_image = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        image = np.array(image)
        results = detect_fire(image)
        st.image(results.render()[0], caption="Hasil Deteksi Kebakaran", use_column_width=True)
