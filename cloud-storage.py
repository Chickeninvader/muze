
import os
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cloud-key.json"

# Set up the client
client = storage.Client()

bucket_name = "muze-train-data"
folder_name = "decoded/"

# Get the bucket
bucket = client.get_bucket(bucket_name)

# TODO: Cloud storage operations

client.close()
