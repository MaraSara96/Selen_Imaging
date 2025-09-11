### This file is identical to api.py apart from the parameters used in the Cellpose model ("diameter=150" and "augment=False"
### were left out in this version)

import os
import base64
import multiprocessing
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from params import *
from PIL import Image
import io
from io import BytesIO
from cellpose import models
from cellpose.io import imread
from model_load import load_model
import matplotlib.pyplot as plt

### Initializing FastAPI:
segmentation = FastAPI()

### CORS Middleware to allow for cross-origin requests from Streamlit:
origins = ["*"]
segmentation.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### Applying segmentation/cropping on uploaded image:
@segmentation.post("/segment/")
async def segment_uploaded_file(file: UploadFile = File(...)):
    all_cells = []
    try:
        ### Initializing Cellpose model for segmentation:
        model = models.CellposeModel(gpu=False, model_type='cyto')

        ### Reading file:
        contents = await file.read()
        image_stream = BytesIO(contents)
        img = Image.open(image_stream)

        ### Converting to RGB if not already the case (e.g. PNG images don't come as RGB):
        if img.mode not in ('RGB'):
            img = img.convert('RGB')

        ### Saving and opening in-between, as imread would only work images opened from a file:
        img.save('temp.jpg', format='JPEG')
        img = imread('temp.jpg')

        ### Applying Cellpose model
        try:
            masks, flows, diams = model.eval(img, channels=[0, 0])

        except Exception as e:
            print(e)
            return e

        ### Applying Cellpose image segments onto image
        try:

            for i in range(1, masks.max()+1):
                mask = masks == i  # Get mask for cell i

                ### Find bounding box for this mask:
                coords = np.argwhere(mask)
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)

                ### Crop original image using bounding box:
                cropped = img[y0:y1+1, x0:x1+1]

                # Optional - apply mask to cropped image (zero out background):
                mask_cropped = mask[y0:y1+1, x0:x1+1]
                cropped_masked = cropped.copy()
                cropped_masked[~mask_cropped] = 0

                all_cells.append(cropped_masked)

        except Exception as e:
            print(e)
            return e

        ### Loading deep learning model, processing segmented image data and preparing data for return to API-calling website
        try:

            prediction_dict = {}

            for index, cell in enumerate(all_cells):

                cell_number = f"Cell {index}"
                pil_image = Image.fromarray(cell)
                resized_pil_image = pil_image.resize((256, 256))
                image_array = np.array(resized_pil_image)
                image_array_with_batch_dim = np.expand_dims(image_array, axis=0)   # Expanding by batch dimension
                model_name = '20250903-091757_ModelTrainedOnSegData_simpleCNN_final.keras' # hard-coded, as Google Cloud Storage bucket access would not work
                my_model = load_model(type="default")
                prediction = my_model.predict(image_array_with_batch_dim)
                class_index = np.argmax(prediction, axis=1)

                ### Processing image date ahead of handover to API-calling website
                byte_io = BytesIO()
                resized_pil_image.save(byte_io, format='JPEG')
                byte_io.seek(0)
                encoded_image = base64.b64encode(byte_io.getvalue()).decode('utf-8')

                # Constructing dictionary for data return to API-calling website
                cell_number = f"Cell {index+1}"
                prediction_dict.setdefault(cell_number, {})
                prediction_dict[cell_number]['image'] = encoded_image
                prediction_dict[cell_number]['class 1'] = f"{float(prediction[0,0]) * 100:.2f}%"
                prediction_dict[cell_number]['class 2'] = f"{float(prediction[0,1]) * 100:.2f}%"
                prediction_dict[cell_number]['class 3'] = f"{float(prediction[0,2]) * 100:.2f}%"
                prediction_dict[cell_number]['class 4'] = f"{float(prediction[0,3]) * 100:.2f}%"
                prediction_dict[cell_number]['class 5'] = f"{float(prediction[0,4]) * 100:.2f}%"
                prediction_dict[cell_number]['class 6'] = f"{float(prediction[0,5]) * 100:.2f}%"
                prediction_dict[cell_number]['class index'] = int(class_index)+1
                prediction_dict[cell_number]['class index probability'] = f"{float(prediction[0,int(class_index)]) * 100:.2f}%"
                prediction_dict[cell_number]['model used'] = model_name

        except Exception as e:
            print(e)
            return e

        return prediction_dict

    except Exception as e:
        print(e)
        return e
