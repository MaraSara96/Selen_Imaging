import os
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from registry import load_model
from pred import *
from params import *

app = FastAPI()
# app.state.model = load_model()

# CORS Middleware to allow for cross-origin requests from Streamlit
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# print("Test Output")

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    try:
        # Uploaded file is saved in a temporary storage space
        with open(f"uploaded_{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Give confirmation
        return {"filename": file.filename, "message": "File received!"}

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    response_example = "This is just a standard API output"
    return response_example
