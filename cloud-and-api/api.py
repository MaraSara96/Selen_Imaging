import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from params import *
from PIL import Image
from io import BytesIO
from model_load import load_model

app = FastAPI()
# app.state.model = load_model()    # This can be activated later to keep the model "alive" while the app is running

# CORS Middleware to allow for cross-origin requests from Streamlit
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/uploadfile/")
async def process_uploaded_file_for_prediction(file: UploadFile):

    try:
        contents = await file.read()
        image_stream = BytesIO(contents)
        image = Image.open(image_stream)
        image = image.resize((256, 256))

        image_processed = np.array(image)
        image_with_batch_dim = np.expand_dims(image_processed, axis=0)

        try:    # check if latest model gives proper output
            my_model = load_model(version="latest")
            prediction = my_model.predict(image_with_batch_dim)

        except:  # use 2nd latest model if latest model does not work
            my_model = load_model(version="fallback")
            prediction = my_model.predict(image_with_batch_dim)

        class_index = np.argmax(prediction, axis=1)   # Determining the predicted class with the highest probability

        prediction_dict = {}
        prediction_dict['class 1'] = f"{float(prediction[0,0]) * 100:.2f}%"
        prediction_dict['class 2'] = f"{float(prediction[0,1]) * 100:.2f}%"
        prediction_dict['class 3'] = f"{float(prediction[0,2]) * 100:.2f}%"
        prediction_dict['class 4'] = f"{float(prediction[0,3]) * 100:.2f}%"
        prediction_dict['class 5'] = f"{float(prediction[0,4]) * 100:.2f}%"
        prediction_dict['class index'] = int(class_index)+1

        return prediction_dict

    except Exception as e:
       return {"error": str(e)}

@app.get("/")
def root():
    response_example = "This is just a standard API output"
    return response_example
