
import os
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cloud-key.json"

# Set up the client
client = storage.Client()

# Set the name of the bucket and the local folder to upload
bucket_name = "muze-train-data"
folder_name = "audio/"

# Get the bucket
bucket = client.get_bucket(bucket_name)

# Upload the folder to the bucket

# TODO: Cloud storage operations

client.close()
