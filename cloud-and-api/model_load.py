import glob
import os
import time

# from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage
from params import *

def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model from GCS
    Return None (but do not Raise) if no model is found

    """
    # print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)
    print(f"Versuche, auf Bucket zuzugreifen: {BUCKET_NAME}")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blobs = client.bucket(BUCKET_NAME).list_blobs(prefix="models/")
    latest_blob = max(blobs, key=lambda x: x.updated)
    # latest_blob_name = latest_blob(blob_name)

    gs_uri = f"gs://{BUCKET_NAME}/{latest_blob.name}"

    # latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
    # latest_blob.download_to_filename(latest_model_path_to_save)
    
    # latest_model = keras.models.load_model(gs_uri)

    latest_model = keras.models.load_model('gs://selen_imaging/models/20250828-083558.keras')   # hardcoded just for the time being for testing

    # print(gs_uri)
    # print(latest_blob.name)
    print("✅ Latest model downloaded from cloud storage")

    return latest_model


    # print(blobs.updated)
    
    # print(blobs.name)

    # blobs = bucket.list_blobs 
    # blobs_list = list(blobs)
    # print(blobs_list)

    

    # try:
        
    #     client = storage.Client()
    #     bucket = client.bucket(BUCKET_NAME)

    #     blobs = bucket.list_blobs     # (prefix="models/")
    #     blobs_list = list(blobs)
    #     print(blobs_list)
    #     # blobs = list(client.get_bucket(BUCKET_NAME)).list_blobs(prefix="models/")
        
        # latest_blob = max(blobs, key=lambda x: x.updated)
    #     # latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
    #     # latest_blob.download_to_filename(latest_model_path_to_save)

    #     # latest_model = keras.models.load_model(latest_model_path_to_save)

    #     print("✅ Latest model downloaded from cloud storage")

    #     return latest_model
    # except:
    #     print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

    #     return None