import glob
import os
import time
from colorama import Fore, Style
from tensorflow import keras
# from google.cloud import storage
from params import *

def load_model(type="latest", version="fallback") -> keras.Model:
    """
    Return a saved model from GCS

    """
    # print(Fore.BLUE + f"\nLoading model from GCS..." + Style.RESET_ALL)

    # client = storage.Client()
    # bucket = client.bucket(BUCKET_NAME)

    # blobs = client.bucket(BUCKET_NAME).list_blobs(prefix="models/")

    # if type == "user_selected":    # get user-selected model
    #     gs_uri_user_selected = f"gs://{BUCKET_NAME}/{version}"
    #     model_user_selected = keras.models.load_model(gs_uri_user_selected)

    #     print("✅ User-selected model downloaded from Google Cloud Storage")
    #     print(f"""Model specifications: {version}""")

    #     return model_user_selected

    if type == "fallback":    # get fallback model

        # blob_fallback = sorted(blobs, key=lambda x: x.updated, reverse=True)[4]  # Just going back 3 model iterations
        # gs_uri_fallback = f"gs://{BUCKET_NAME}/{blob_fallback.name}"

        model_name = '20250903-091757_ModelTrainedOnSegData_simpleCNN_final.keras'   # hard-coded for the time being

        model_fallback = keras.models.load_model(model_name)

        print(f"""Model loaded: {model_name}""")

        return model_fallback
