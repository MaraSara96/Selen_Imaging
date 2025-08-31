import streamlit as st
import datetime
import requests
import io
import pandas as pd
from google.cloud import storage
from params import *
from PIL import Image

# API_URL = st.secrets["API_URL"]    # API_URL stored in a local "secrets" file; in production, API_URL will be stored in Streamlit's secrets section in the web interface
API_URL = 'http://127.0.0.1:8000/uploadfile/'  # API URL hardcoded to the local server for the time being


st.set_page_config(page_title="Leukemia Predictor", page_icon="🩸", layout="centered")
st.title("🩸 Leukemia Image Classification (MVP)")
st.caption("Upload a microscope image → API → prediction")

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
blobs_list = list(bucket.list_blobs(prefix="models/"))
sorted_blobs = sorted(blobs_list, key=lambda x: x.updated, reverse=True)
list_of_blobs = [blob.name for blob in sorted_blobs]
selected_model = st.selectbox("**Choose the model to be used**" + "\n\n" + "*(list sorted descending by model deployment date - latest model is preselected)*", list_of_blobs)
data_to_send = {"option": selected_model}

file = st.file_uploader("Upload PNG/JPG" + "\n\n" +":red[(Only upload B/W images in combination with a '... BW' model selection above. Processing B/W images in models trained on color images might lead to wrong classifications.)]", type=["png","jpg","jpeg"])


if file:
    image = Image.open(file)
    st.image(image, caption="Preview", use_container_width=True)
    image_mode = image.mode
    image_format = image.format

    if "BW" in selected_model:
        image = image.convert('L')   # Converting to greyscale in any case, as detection via image.mode is not always reliable
        jpg_stream = io.BytesIO()
        image.save(jpg_stream, format='JPEG')
        user_feedback = "*Note: You selected a model trained on greyscale images. Your uploaded image was automatically converted to greyscale (if not already the case).*"

    else:
        if image.mode in ('1','L'):
            user_feedback = "*Note: You uploaded a BW / greyscale image. Note that your selected model may have been trained on color images.*"

        else:
            if image.format in ('PNG','png'):
                image = image.convert('RGB')
                jpg_stream = io.BytesIO()
                image.save(jpg_stream, format='JPEG')
                user_feedback = "*Note: Uploaded PNG image was automatically converted to JPG.*"
            else:
                user_feedback = "Image processed."

    if st.button("Predict"):
        status_placeholder = st.empty()
        status_placeholder.text("Processing... please stand by.")
        buf = io.BytesIO(); image.save(buf, format="JPEG"); buf.seek(0)

        try:
            r = requests.post(API_URL, files={"file": ("image.jpg", buf, "image/jpeg")}, data=data_to_send, timeout=20)
            r.raise_for_status()
            data = r.json()
            status_placeholder.markdown(user_feedback)
            st.write(f"""##### This image belongs to class **{data['class index']}**.""")
            st.write(f"###### Probabilities:")
            st.text(f"Class 1 (Basophil): {data['class 1']} \nClass 2 (Erythroblast): {data['class 2']} \nClass 3 (Monocyte): {data['class 3']} \nClass 4 (Myeloblast): {data['class 4']} \nClass 5 (Seg Neutrophil): {data['class 5']}")

        except Exception as e:
           st.error(f"API error: {e}")
else:
    st.info("Please upload an image to start.")
