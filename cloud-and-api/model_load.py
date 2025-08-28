import glob
import os
import time

from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage
from params import *

def load_model(version="latest") -> keras.Model:
    """
    Return a saved model from GCS

    """
    print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blobs = client.bucket(BUCKET_NAME).list_blobs(prefix="models/")

    if version == "latest":    # get latest model
        latest_blob = max(blobs, key=lambda x: x.updated)
        gs_uri_latest = f"gs://{BUCKET_NAME}/{latest_blob.name}"
        latest_model = keras.models.load_model(gs_uri_latest)

        print("✅ Latest model downloaded from Google Cloud Storage")
        print(f"""Model specifications: {latest_blob.name}""")

        return latest_model

    # get 2nd latest model
    latest_blob_2nd = sorted(blobs, key=lambda x: x.updated, reverse=True)[1]
    gs_uri_2ndlatest = f"gs://{BUCKET_NAME}/{latest_blob_2nd.name}"
    latest_model_2nd = keras.models.load_model(gs_uri_2ndlatest)

    print("✅ Fallback model downloaded from Google Cloud Storage")
    print(f"""Model specifications: {latest_blob_2nd.name}""")

    return latest_model_2nd
