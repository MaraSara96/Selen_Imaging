import glob
import os
import time
from colorama import Fore, Style
from tensorflow import keras
# from google.cloud import storage   # deactivated as Google Cloud Storage bucket access would not work
from params import *

### Loading model (either locally stored default/fallback model or user-selected model)
def load_model(type="default", version="20250903-091757_ModelTrainedOnSegData_simpleCNN_final.keras") -> keras.Model:
    """
    Return a saved model from GCS or use a default one (available as a file locally)

    """

    ### Following section deactivated as Google Cloud Storage bucket access would not work
    # print(Fore.BLUE + f"\nLoading model from GCS..." + Style.RESET_ALL)
    # client = storage.Client()
    # bucket = client.bucket(BUCKET_NAME)
    # blobs = client.bucket(BUCKET_NAME).list_blobs(prefix="models/")

    if type == "user_selected":   # get user-selected model (specific model can be handed over via 'version' variable)
        gs_uri_user_selected = f"gs://{BUCKET_NAME}/{version}"
        model_user_selected = keras.models.load_model(gs_uri_user_selected)

        print("✅ User-selected model downloaded from Google Cloud Storage")
        print(f"""Model specifications: {version}""")

        return model_user_selected

    if type == "default":   # get default model
        model_name = '20250903-091757_ModelTrainedOnSegData_simpleCNN_final.keras'  # hard-coded, as Google Cloud Storage bucket access would not work
        model_default = keras.models.load_model(model_name)
        print(f"""Model loaded: {model_name}""")

        return model_default
