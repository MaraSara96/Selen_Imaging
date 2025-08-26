import io, requests, streamlit as st
from PIL import Image

st.set_page_config(page_title="Leukemia Predictor", page_icon="🩸", layout="centered")
st.title("🩸 Leukemia Image Classification (MVP)")
st.caption("Upload a microscope image → API → prediction")

# API_URL = st.secrets.get("API_URL", "http://localhost:8000/predict")
API_URL = "https://selen-imagingtorstenweindl-248422586834.europe-west1.run.app/"

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
            prob = float(data.get("probability", 0.0))
            st.success(f"Result: **{data.get('label','?').upper()}** · "
                       f"Confidence: {prob:.2%} · "
                       f"Model: {data.get('model_version','?')}")
            st.progress(min(max(prob,0.0),1.0))
        except Exception as e:
            st.error(f"API error: {e}")
else:
    st.info("Please upload an image to start.")
