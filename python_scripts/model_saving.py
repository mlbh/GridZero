import os

from google.cloud import storage
import joblib

# # saving model
# grid1_xgb = grid.best_estimator_
# # make folder
# MODEL_DIR = os.path.join('..', 'models_mlbh')
# os.makedirs(MODEL_DIR, exist_ok=True)
# # choose file path
# MODEL_PATH = os.path.join(MODEL_DIR, 'grid1_xgb.json')
# # save model
# best_xgb.save_model(MODEL_PATH)

# print('Saved to:', MODEL_PATH)


# # saving to GCP bucket - bucket_name = 'grid_zero_bucket'
# bucket_name = 'grid_zero_bucket'
# blob_name = 'grid1_xgb.json'

# MODEL_PATH = '../models_mlbh/grid1_xgb.json'

# client = storage.Client()
# bucket = client.bucket(bucket_name)
# blob = bucket.blob(blob_name)

# blob.upload_from_filename(MODEL_PATH)

# print(f'Uploaded to gs://{bucket_name}/{blob_name}')


# function
import os
from google.cloud import storage

def save_and_upload_to_gcp(
        model,
        local_dir='../models_mlbh',
        model_filename='type_model_v_ddmm.json',
        bucket_name='grid_zero_bucket',
        blob_name=None
    ):
    '''
    Save trained XGBoost model locally and upload to GCP.
    '''

    # create local directory
    os.makedirs(local_dir, exist_ok=True)
    model_path = os.path.join(local_dir, model_filename)

    # save locally
    model.save_model(model_path)
    print(f'Model saved locally to {model_path}')

    # upload to GCP
    if blob_name is None:
        blob_name = model_filename

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(model_path)

    print(f'Uploaded to gs://{bucket_name}/{blob_name}')

    return model_path


def save_and_upload_anything(
        obj,
        filename,
        local_dir="../models_mlbh",
        bucket_name="grid_zero_bucket",
        blob_name=None
        ):

    os.makedirs(local_dir, exist_ok=True)
    path = os.path.join(local_dir, filename)

    # save object
    joblib.dump(obj, path)
    print(f"Saved locally → {path}")

    # upload
    if blob_name is None:
        blob_name = filename

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path)

    print(f"Uploaded → gs://{bucket_name}/{blob_name}")

    return path
