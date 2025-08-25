import pandas as pd
from fastapi import FastAPI, UploadFile, File
from registry import load_model
from pred import pred

app = FastAPI()
app.state.model = load_model()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename, "content_type": file.content_type}
