import os
import streamlit as st
import datetime
import requests
from IPython.display import display
import io
from collections import Counter
import base64
import math
import pandas as pd
from google.cloud import storage
from params import *
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

st.set_page_config(
    page_title="Leukemia Image Classification (MVP)",
    page_icon="🩸",
    # layout="wide",
    initial_sidebar_state="collapsed",
    # menu_items={}
)

# API_URL = st.secrets["API_URL"]    # API_URL stored in a local "secrets" file; in production, API_URL will be stored in Streamlit's secrets section in the web interface
# API_URL = 'http://127.0.0.1:8000/segment/'  # API URL hardcoded to the local server for the time being
API_URL = st.secrets['API_URL_TESTING']
BUCKET_NAME = st.secrets['BUCKET_NAME']
# GCP_PROJECT = st.secrets['GCP_PROJECT']


model_list = ['20250826-152119.keras','20250827-141827.keras','20250828-082537.keras','20250828-083337.keras','20250828-083558.keras','20250828-154636_model BW.keras','20250829-130807_model BW.keras',
              '20250902-193012_model BW.keras','20250902-193056_ModelTrainedOnSegData_simpleCNN_v1.keras','20250903-073842_ModelTrainedOnSegData_simpleCNN_v1.keras','20250903-091757_ModelTrainedOnSegData_simpleCNN_final.keras']

# client = storage.Client()
# bucket = client.bucket(BUCKET_NAME)
# blobs_list = list(bucket.list_blobs(prefix="models/"))
# sorted_blobs = sorted(blobs_list, key=lambda x: x.updated, reverse=True)
# list_of_blobs = [blob.name for blob in sorted_blobs]
# selected_model = st.selectbox("**Choose the model to be used**" + "\n\n" + "*(list sorted descending by model deployment date - latest model is preselected)*", list_of_blobs)
# data_to_send = {"option": selected_model}

# data_to_send = {"option_1": "no value selected", "option_2": "no value selected", "option_3": "no value selected"}
st.set_page_config(page_title="Leukemia Image Classification (MVP)", page_icon="🩸", layout="centered")
st.title("🩸 Leukemia Image Classification (MVP)")
st.caption("L.-P. Abensour, Z. Awad, L. Bird, M. Sarateanu, T. Weindl")
st.caption("Upload a blood smear image and have our API classify the different blood cells. **Please don't use B/W images, as model was trained on color images.**")
file = st.file_uploader("Upload JPG/PNG of a blood smear", type=["png","jpg","jpeg"])

if file:
  image = Image.open(file)
  if image.mode != 'RGB':
    image = image.convert('RGB')
  buf = io.BytesIO(); image.save(buf, format="JPEG"); buf.seek(0)
  original_width, original_height = image.size

  if original_height > 500:
    image_ratio = original_width / original_height
    preview_image = image.resize((int(500*image_ratio),500))
  else:
    preview_image = image
  st.image(preview_image, caption="Preview")    # replaced "use_container_width=True"

  if st.button("Start detection"):
    status_placeholder = st.empty()
    status_placeholder.text(f"""8 mighty CPUs are digging into it right now ;) - please stand by for about 1 minute (maybe a little more for heavy images).\nOnce processed, the results will be shown below.""")

    try:
      r = requests.post(API_URL, files={"file": ("image.jpg", buf, "image/jpeg")}, timeout=600)
      r.raise_for_status()
      status_placeholder.text(f"Detection done.")
      data = r.json()

      class_mapping = {
      1: 'Basophil',
      2: 'Erythroblast',
      3: 'Monocyte',
      4: 'Myeloblast',
      5: 'Seg Neutrophil',
      6: 'Red Blood Cell'
      }

      for cell in data:
        inner_dict = data[cell]
        original_value = inner_dict['class index']
        recoded_value = class_mapping.get(original_value, 'Unknown')
        inner_dict['class index'] = recoded_value

      classes = [inner_dict['class index'] for inner_dict in data.values()]
      classes_count = Counter(classes)
      classes_count_sorted = sorted(classes_count.items(), key=lambda item: item[1], reverse=True)

      total_items = len(data)
      st.write(f"#### We found {total_items} cells:")

      bullet_list = ""
      for i in classes_count_sorted:
        bullet_list += f"- <b>{i[1]} x {i[0]}</b> ({float(i[1]) / total_items * 100:.2f}% of all cells)<br>"

      st.markdown(f"<span style='font-size:1.2em;'> {bullet_list}", unsafe_allow_html=True)

      myeloblast_count = 0
      for cell_data in data.values():
        if cell_data.get('class index') == 'Myeloblast':
          myeloblast_count += 1

      if myeloblast_count > 1:
        st.markdown("<span style='color:red; font-size:1.3em;'><b>" + str(myeloblast_count) + " Myeloblast cell(s) found, which can indicate blood cancer.</b>", unsafe_allow_html=True)
      else:
        st.write(f"No direct indication for blood cancer from this blood smear.")

      st.markdown("<br>", unsafe_allow_html=True)

      st.markdown(f"#### Blood cell types in focus:<span style='font-size:0.6em;'><br>(in descending order of detection confidence)", unsafe_allow_html=True)


      dict_wo_rbc = {key: values for key, values in data.items() if values['class index'] != "Red Blood Cell"}
      dict_rbc = {key: values for key, values in data.items() if values['class index'] == "Red Blood Cell"}

      confidence_order = sorted(dict_wo_rbc.items(), key=lambda item: item[1]['class index probability'], reverse=True)
      dict_wo_rbc_by_confidence = {key: value for key, value in confidence_order}

      total_items_wo_rbc = len(dict_wo_rbc)
      total_items_rbc = len(dict_rbc)

	  # Table with 'interesting' blood cells
      num_columns_wo_rbc = 3
      data_list_wo_rbc = list(dict_wo_rbc_by_confidence.items())
      num_rows_wo_rbc = math.ceil(total_items_wo_rbc / num_columns_wo_rbc)

      for row in range(num_rows_wo_rbc):
        cols_wo_rbc = st.columns(num_columns_wo_rbc)
        for col_index in range(num_columns_wo_rbc):
          item_index_wo_rbc = row * num_columns_wo_rbc + col_index
          if item_index_wo_rbc < total_items_wo_rbc:
            key, value = data_list_wo_rbc[item_index_wo_rbc]
            with cols_wo_rbc[col_index]:
              st.write(f"##### {key}")
              binary_data_wo_rbc = base64.b64decode(value["image"])
              image_stream_wo_rbc = io.BytesIO(binary_data_wo_rbc)
              st.image(image_stream_wo_rbc)
              st.write(f"""**{value["class index"]}**""")
              st.write(f"""(confidence: **{value["class index probability"]}**)""")

      st.markdown("<br>", unsafe_allow_html=True)
      st.write(f"#### Regular red blood cells:")

	  # Table with red blood cells
      num_columns_rbc = 3
      data_list_rbc = list(dict_rbc.items())
      num_rows_rbc = math.ceil(total_items_rbc / num_columns_rbc)

      for row in range(num_rows_rbc):
        cols_rbc = st.columns(num_columns_rbc)
        for col_index in range(num_columns_rbc):
          item_index_rbc = row * num_columns_rbc + col_index
          if item_index_rbc < total_items_rbc:
            key, value = data_list_rbc[item_index_rbc]
            with cols_rbc[col_index]:
              st.write(f"##### {key}")
              binary_data_rbc = base64.b64decode(value["image"])
              image_stream_rbc = io.BytesIO(binary_data_rbc)
              st.image(image_stream_rbc)
              st.write(f"""**{value["class index"]}**""")
              st.write(f"""(confidence: **{value["class index probability"]}**)""")

      st.markdown("<br><br><br>", unsafe_allow_html=True)
      st.write(f"""**Model used for detection:** {data['Cell 1']['model used']}""")

    except Exception as e:
         st.error(f"API error: {e}")
else:
    st.info("Please upload an image to start.")
