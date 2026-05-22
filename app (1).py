import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(page_title="🐦 Bird Detector", layout="centered")
st.title("🐦 Bird Detection using YOLOv8")
st.markdown("Upload an image to detect birds using a fine-tuned YOLOv8n model.")

WEIGHTS_PATH = 'best.pt'

if not os.path.exists(WEIGHTS_PATH):
    st.error("⚠️ Model weights not found. Make sure 'best.pt' is in the same folder.")
else:
    model = YOLO(WEIGHTS_PATH)
    uploaded_file = st.file_uploader("Choose a bird image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        if st.button('🔍 Run Bird Detection'):
            with st.spinner('Detecting birds...'):
                results = model.predict(source=image, conf=0.25)
                res_plotted = results[0].plot()[:, :, ::-1]
                n_birds = len(results[0].boxes)

            st.image(res_plotted, caption=f'Detection Result — {n_birds} bird(s) found',
                     use_container_width=True)

            if n_birds > 0:
                st.success(f"✅ Detected {n_birds} bird(s) in the image!")
                for i, box in enumerate(results[0].boxes):
                    conf = box.conf.item()
                    st.write(f"  Bird {i+1}: Confidence = {conf:.2%}")
            else:
                st.warning("⚠️ No birds detected. Try a different image or lower the confidence threshold.")
