from google.cloud import storage
from pathlib import Path

class GCSModelLoader:

    def __init__(self, bucket_name: str, local_dir: str = "./models"):
        self.bucket_name = bucket_name
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(exist_ok=True)

        self.client = storage.Client(project="gridzero-489711")
        self.bucket = self.client.bucket(bucket_name)

    def download_file(self, gcs_path: str):
        """
        Download file from GCS if not already cached locally
        """
        filename = Path(gcs_path).name
        local_path = self.local_dir / filename

        if local_path.exists():
            return local_path

        blob = self.bucket.blob(gcs_path)
        blob.download_to_filename(local_path)

        return local_path
