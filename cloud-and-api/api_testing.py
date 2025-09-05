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

segmentation = FastAPI()

# CORS Middleware to allow for cross-origin requests from Streamlit
origins = ["*"]
segmentation.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Applying segmentation on uploaded image
@segmentation.post("/segment/")
async def segment_uploaded_file(file: UploadFile = File(...)):
    all_cells = []
    print("all cells" + str(all_cells))
    try:
        # Initializing Cellpose model for segmentation
        model = models.CellposeModel(gpu=False, model_type='cyto')

        contents = await file.read()
        image_stream = BytesIO(contents)
        img = Image.open(image_stream)    # Saving in-between, as resizing on the go produced an error
        # img_resized = img.resize((50, 50))

        if img.mode not in ('RGB'):
            img = img.convert('RGB')

        img.save('temp.jpg', format='JPEG')
        img = imread('temp.jpg')

        try:
            # Diameter not specified (takes more computation), augmentation on (takes more computation)
            # masks, flows, diams = model.eval(img, channels=[0, 0], diameter=None, augment=True)

            masks, flows, diams = model.eval(img, channels=[0, 0])

        except Exception as e:
            print(e)
            return e

        try:

            for i in range(1, masks.max()+1):
                mask = masks == i  # Get mask for cell i

                # Find bounding box for this mask
                coords = np.argwhere(mask)
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)

                # print("x0" + x0)

                # Crop original image using bounding box
                cropped = img[y0:y1+1, x0:x1+1]

                # Optional: apply mask to cropped image (zero out background)
                mask_cropped = mask[y0:y1+1, x0:x1+1]
                cropped_masked = cropped.copy()
                cropped_masked[~mask_cropped] = 0

                all_cells.append(cropped_masked)

        except Exception as e:
            print(e)
            return e


        try:    # load model selected by user

            prediction_dict = {}

            for index, cell in enumerate(all_cells):

                cell_number = f"Cell {index}"
                # print(cell_number)
                pil_image = Image.fromarray(cell)
                # print("Image read von array")
                resized_pil_image = pil_image.resize((256, 256))
                # print("Image resized")
                image_array = np.array(resized_pil_image)
                # print("Image reverted to array")
                image_array_with_batch_dim = np.expand_dims(image_array, axis=0)
                # print("Epanding batch dimension")
                model_name = '20250903-091757_ModelTrainedOnSegData_simpleCNN_final.keras' # hard-coded for the time being
                my_model = load_model(type="fallback")
                # print("Loading model")
                prediction = my_model.predict(image_array_with_batch_dim)
                # print("Predicting")
                class_index = np.argmax(prediction, axis=1)
                # print("Determining class_index")

                byte_io = BytesIO()
                resized_pil_image.save(byte_io, format='JPEG')
                byte_io.seek(0)
                encoded_image = base64.b64encode(byte_io.getvalue()).decode('utf-8')

                cell_number = f"Cell {index+1}"
                prediction_dict.setdefault(cell_number, {})
                # print("First dict key set")
                prediction_dict[cell_number]['image'] = encoded_image
                # print("image array written into dict")
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

        # prediction_dict = {}
        # prediction_dict['class 1'] = f"{float(prediction[0,0]) * 100:.2f}%"
        # prediction_dict['class 2'] = f"{float(prediction[0,1]) * 100:.2f}%"
        # prediction_dict['class 3'] = f"{float(prediction[0,2]) * 100:.2f}%"
        # prediction_dict['class 4'] = f"{float(prediction[0,3]) * 100:.2f}%"
        # prediction_dict['class 5'] = f"{float(prediction[0,4]) * 100:.2f}%"
        # prediction_dict['class 6'] = f"{float(prediction[0,5]) * 100:.2f}%"
        # prediction_dict['class index'] = int(class_index)+1

        return prediction_dict

    except Exception as e:
        print(e)
        return e
