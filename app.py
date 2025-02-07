import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load model YOLOv5m yang sudah dilatih
MODEL_PATH = "yolov5m_wildfire.pt"  # Ganti dengan path model Anda
model = YOLO(MODEL_PATH)

# Fungsi untuk melakukan deteksi
def detect_fire(image):
    results = model(image)
    detections = results[0].boxes.data.cpu().numpy()  # Mengambil bounding box

    for box in detections:
        x1, y1, x2, y2, conf, cls = box
        if conf > 0.5:  # Tampilkan hanya jika confidence > 50%
            label = f"WILDFIRE {conf:.2f}"
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image

# Konfigurasi UI Streamlit
st.title("🔥 Deteksi Kebakaran Hutan Real-Time")
st.write("Gunakan kamera untuk mendeteksi kebakaran hutan secara real-time.")

# Pilih sumber input: Kamera atau Upload Gambar
option = st.radio("Pilih Sumber Input:", ("Kamera", "Upload Gambar"))

if option == "Kamera":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Gagal membaca kamera.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_frame = detect_fire(frame)

        stframe.image(detected_frame, channels="RGB", use_column_width=True)

    cap.release()

elif option == "Upload Gambar":
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        detected_image = detect_fire(image)
        st.image(detected_image, channels="RGB", use_column_width=True)
