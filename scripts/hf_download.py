from datasets import load_dataset
import os
import soundfile as sf
import numpy as np
from ezspeech.utils.common import save_dataset
# Load the dataset
dataset = load_dataset("Khanh14ph/asr-youtube-dataset")

# Create output directory
dataset_name = "youtube"    
output_dir = f"/scratch/midway3/khanhnd/data/audio/{dataset_name}"
os.makedirs(output_dir, exist_ok=True)
lst=[]
# Save audio files
for idx, sample in enumerate(dataset["train"]):
    
    # Create filename
    filename = f"audio_{idx:05d}.flac"
    filepath = os.path.join(output_dir, filename)
    if idx % 1000 == 0:
        print(f"Saved: {filename}")
    if os.path.exists(filepath):
        lst.append({"audio_filepath": f"{dataset_name}/{filename}", "text": sample["transcript"],"duration": len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]})
        continue
    # Get audio data
    audio_data = sample["audio"]
    audio_array = audio_data["array"]
    sample_rate = audio_data["sampling_rate"]
    
    
    
    # Save audio file
    sf.write(filepath, audio_array, sample_rate)
    lst.append({"audio_filepath": f"{dataset_name}/{filename}", "text": sample["transcript"],"duration": len(audio_array) / sample_rate})
    
    
    # Optional: limit number of files for testing
    # if idx >= 10:  # Remove this line to save all files
    #     break
    
print(f"Audio files saved to: {output_dir}")
save_dataset(lst, f"/scratch/midway3/khanhnd/data/metadata/{dataset_name}.jsonl")