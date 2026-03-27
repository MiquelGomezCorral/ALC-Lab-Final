
import os

from maikol_utils.file_utils import list_dir_files, load_json, save_json
from maikol_utils.file_utils import print_error
from tqdm import tqdm

from transformers import pipeline
import warnings

from src.config import Configuration

def get_extra_sounds(CONFIG: Configuration) -> str:
    """
    Placeholder function to get extra sounds from a video.
    In a real implementation, this would use audio classification models.
    
    Args:
        CONFIG (Configuration): The configuration object containing paths and settings.
    Returns:
        str: The extra sounds detected in the video.
    """
    # files, n = list_dir_files(CONFIG.videos_path)#, max_files=10)
    metadata = load_json(CONFIG.videos_data)


    hf_device = 0 #if device == "cuda" else -1 

    # Loads HuBERT for tone/emotion (neutral, happy, angry, sad)
    emotion_model = pipeline("audio-classification", model="superb/hubert-large-superb-er", device=hf_device)

    # Loads AudioSpectrogramTransformer for 527 background sounds (music, laughter, sirens, etc.)
    event_model = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593", device=hf_device)

    # Loads Wav2Vec2 for binary gender detection (male/female)
    gender_model = pipeline("audio-classification", model="prithivMLmods/Common-Voice-Gender-Detection", device=hf_device)


    features = {}
    errors = []
    for data in tqdm(metadata.values(), desc="Processing videos"):
        video_path = os.path.join(CONFIG.videos_path, data["video"])

        try:

            tone, events, gender = extract_audio_cues(
                video_path,
                emotion_model,
                event_model,
                gender_model
            )
            features[data["video"]] = {"tone": tone, "background_sounds": events, "gender": gender}
        except Exception as e:
            print_error(f"Error occurred while processing {video_path}: {e}")
            errors.append(video_path)
            features[data["video"]] = {"tone": None, "background_sounds": None, "gender": None}
            continue

    if errors:
        print_error(f"Errors occurred for {len(errors)} videos: {errors}")

    save_json(CONFIG.extra_sounds_path, features)


def extract_audio_cues(video_path, emotion_model, event_model, gender_model):
    # These models strictly require 16kHz audio format
    # audio_tensor, sr = torchaudio.load(video_path)
    # audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sr, new_freq=16000)
    # audio = audio_tensor.mean(dim=0).numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        audio, _ = librosa.load(video_path, sr=16000)
        
    # Extract the dominant tone
    emotion_result = emotion_model(audio, top_k=5)
    tones = [res['label'] for res in emotion_result if res['score'] > 0.15]
    tone = ", ".join(tones) if tones else "neutral"
    
    # Extract all background sounds with > 10% confidence
    event_result = event_model(audio, top_k=10)
    background_sounds = [res['label'] for res in event_result if res['score'] > 0.10]
    background_sounds = ", ".join(background_sounds) if background_sounds else "none"
    
    # Extract Gender
    gender_result = gender_model(audio, top_k=1)
    gender = gender_result[0]['label']
    
    return tone, background_sounds, gender