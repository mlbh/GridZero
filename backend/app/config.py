import os

GCS_BUCKET = os.getenv("MODEL_BUCKET", "energy-model-bucket")

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
