import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np

st.set_page_config(page_title="Bird Detector", layout="centered")

st.title("🐦 Bird Detection using YOLOv8")
st.write("Upload an image to detect birds.")

WEIGHTS_PATH = "best.pt"

if not os.path.exists(WEIGHTS_PATH):
    st.error("⚠️ best.pt model file not found.")
else:
    model = YOLO(WEIGHTS_PATH)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Run Detection"):

            with st.spinner("Detecting birds..."):

                results = model.predict(
                    source=np.array(image),
                    conf=0.25
                )

                plotted = results[0].plot()[:, :, ::-1]

                bird_count = len(results[0].boxes)

            st.image(
                plotted,
                caption=f"Detected {bird_count} bird(s)",
                use_container_width=True
            )

            if bird_count > 0:
                st.success(f"✅ Found {bird_count} bird(s)!")

                for i, box in enumerate(results[0].boxes):
                    conf = float(box.conf[0])
                    st.write(f"Bird {i+1}: {conf:.2%} confidence")

            else:
                st.warning("No birds detected.")
