import datetime
import os
import json
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cloud-key.json"

client = storage.Client()
bucket = client.get_bucket("muze-train-data")

with open("train.json", "r") as file:
    train_data = json.load(file)

new_data = {}

for key in train_data.keys():
    blob = bucket.blob(key)
    if blob.exists():
        url = blob.generate_signed_url(
            expiration=datetime.timedelta(hours=1), method='GET')
        new_data[key] = {
            "description": train_data[key],
            "url": url
        }
        print(key, "done")

with open("new_train.json", "w") as file:
    json.dump(new_data, file)
