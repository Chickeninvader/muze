import librosa
from IPython.display import Audio, display, clear_output
import ipywidgets as widgets

# Get the file path to an included audio example
filename = "we_wish_you.mp3"

# Load the audio as a waveform `y`
# Store the sampling rate as `sr`
audio_data, sample_rate = librosa.load(filename)

# Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sample_rate)

# Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)