import os
from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage
from params import *


def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model from GCS
    Return None (but do not Raise) if no model is found

    """

    print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

    client = storage.Client()
    blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

    try:
        latest_blob = max(blobs, key=lambda x: x.updated)
        # latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
        # latest_blob.download_to_filename(latest_model_path_to_save)

        latest_model = keras.models.load_model(latest_blob)

        print("✅ Latest model downloaded from Google Cloud Storage")

        return latest_model

    except:
        print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

        return None


def save_model(model: keras.Model = None) -> None:
    """
    Save model to GCS
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    model.save(model_path)

    model_filename = model_path.split("/")[-1]
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_path)

    print("✅ Model saved to GCS")

    return None
