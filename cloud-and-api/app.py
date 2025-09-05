import os
import streamlit as st
import datetime
import requests
import io
import pandas as pd
from google.cloud import storage
from params import *
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# API_URL = st.secrets["API_URL"]    # API_URL stored in a local "secrets" file; in production, API_URL will be stored in Streamlit's secrets section in the web interface
# API_URL = 'http://127.0.0.1:8000/segment/'  # API URL hardcoded to the local server for the time being
API_URL = os.environ.get("API_URL")

file = st.file_uploader("Upload JPG/PNG", type=["png","jpg","jpeg"])

if file:

    if st.button("Segment"):
        status_placeholder = st.empty()
        status_placeholder.text("Segmenting in the works ... please stand by.")
        image = Image.open(file)
        buf = io.BytesIO(); image.save(buf, format="JPEG"); buf.seek(0)

        try:
            r = requests.post(API_URL, files={"file": ("image.jpg", buf, "image/jpeg")}, timeout=600)
            r.raise_for_status()
            data = r.json()
            st.write("This is the test output:")
            st.write(data)
            # plt.imshow(data[1])
            # plt.show()

        except Exception as e:
           st.error(f"API error: {e}")
else:
    st.info("Please upload an image to start.")
