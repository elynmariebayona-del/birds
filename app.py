import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np

st.set_page_config(page_title="🐦 Bird Detector", layout="centered")
st.title("🐦 Bird Detection using YOLOv8")
st.write("Upload an image to detect birds using a fine-tuned YOLOv8 model.")

# Load model — checks both Streamlit Cloud path and Google Drive path
WEIGHTS_OPTIONS = [
    "best.pt",                                              # Streamlit Cloud (repo root)
    "/content/drive/MyDrive/yolov8_bird_detector_best.pt" # Google Drive / Colab
]

model = None
for path in WEIGHTS_OPTIONS:
    if os.path.exists(path):
        model = YOLO(path)
        break

if model is None:
    st.error("⚠️ Model file not found. Make sure 'best.pt' is in the repo root.")
    st.stop()

uploaded_file = st.file_uploader("Choose a bird image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Run Detection"):
        with st.spinner("Detecting birds..."):
            results = model.predict(source=np.array(image), conf=0.25)
            plotted = results[0].plot()[:, :, ::-1]
            bird_count = len(results[0].boxes)

        st.image(plotted, caption=f"Detected {bird_count} bird(s)", use_container_width=True)

        if bird_count > 0:
            st.success(f"✅ Found {bird_count} bird(s)!")
            for i, box in enumerate(results[0].boxes):
                conf = float(box.conf[0])
                st.write(f"Bird {i+1}: {conf:.2%} confidence")
        else:
            st.warning("⚠️ No birds detected. Try another image.")
