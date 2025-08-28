import streamlit as st
import datetime
import requests
import io
import pandas as pd
from PIL import Image

# API_URL = st.secrets["API_URL"]    # API_URL stored in a local "secrets" file; in production, API_URL will be stored in Streamlit's secrets section in the web interface
API_URL = 'http://127.0.0.1:8000/uploadfile/'  # API URL hardcoded to the local server for the time being

st.set_page_config(page_title="Leukemia Predictor", page_icon="🩸", layout="centered")
st.title("🩸 Leukemia Image Classification (MVP)")
st.caption("Upload a microscope image → API → prediction")

file = st.file_uploader("Upload PNG/JPG", type=["png","jpg","jpeg"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Preview", use_container_width=True)

    if st.button("Predict"):
        buf = io.BytesIO(); img.save(buf, format="JPEG"); buf.seek(0)
        try:
            r = requests.post(API_URL, files={"file": ("image.jpg", buf, "image/jpeg")}, timeout=20)
            r.raise_for_status()
            data = r.json()
            st.write(f"""#### This image belongs to class **{data['class index']}**.""")
            st.write(f"#### Probabilities:")
            st.write(f"""Class 1: {data['class 1']} ---
                     Class 2: {data['class 2']} ---
                     Class 3: {data['class 3']} ---
                     Class 4: {data['class 4']} ---
                     Class 5: {data['class 5']}""")


        except Exception as e:
           st.error(f"API error: {e}")
else:
    st.info("Please upload an image to start.")
