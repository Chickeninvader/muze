import os
import json
import librosa
import gc
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cloud-key.json"

client = storage.Client()
bucket = client.get_bucket("muze-train-data")

with open("train.json", "r") as file:
    train_data = json.load(file)


def process_batch(keys):
    batch_data = {}
    for key in keys:
        blob = bucket.blob(key)
        local_file_path = f"tmp/{key}"
        blob.download_to_filename(local_file_path)
        audio_data, sample_rate = librosa.load(local_file_path)
        tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)

        batch_data[key] = {
            "description": train_data[key],
            "audio_data": audio_data.tolist(),
            "tempo": tempo,
            "beat_frames": beat_frames.tolist(),
            "beat_times": beat_times.tolist(),
        }
        print(f"Decoded {key}")
        os.remove(local_file_path)
    return batch_data


# Define batch size
batch_size = 10  # Reduce batch size

# Split keys into batches
keys = list(train_data.keys())
batches = [keys[i : i + batch_size] for i in range(0, len(keys), batch_size)]

# Process each batch
master_data = {}
for batch in batches:
    batch_data = process_batch(batch)
    master_data.update(batch_data)  # Append batch data to master data
    del batch_data  # Free up memory
    gc.collect()  # Force garbage collection

# Write master data to file
with open("tmp/decoded_data.json", "w") as file:
    json.dump(master_data, file, indent=4)

# Upload to cloud
blob = bucket.blob("decoded/decoded_data.json")
blob.upload_from_filename("tmp/decoded_data.json")
