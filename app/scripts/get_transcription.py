
from maikol_utils.file_utils import list_dir_files, save_json
from maikol_utils.file_utils import print_error
from tqdm import tqdm
import whisperx

from src.config import Configuration

def get_transcriptions(CONFIG: Configuration) -> str:
    """
    Placeholder function to get transcription from a video.
    In a real implementation, this would use a speech-to-text model or service.
    
    Args:
        CONFIG (Configuration): The configuration object containing paths and settings.
    Returns:
        str: The transcription of the video.
    """
    files, n = list_dir_files(CONFIG.videos_path)#, max_files=10)

    device = "cuda" 
    compute_type = "float32" 
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)

    transcriptions = {}
    errors = []
    for f in tqdm(files, desc="Processing videos"):
        transcription = extract_transcription(model, f, CONFIG.batch_size)
        try:
            transcriptions[f] = transcription[0]["text"]
        except IndexError:
            transcriptions[f] = ""
            print_error(f"Transcription for {f} is empty. Setting it to an empty string.")
            errors.append(f)

    if errors:
        print_error(f"Errors occurred for {len(errors)} videos: {errors}")

    save_json(transcriptions, CONFIG.transcriptions_path)


def extract_transcription(model, video_path, batch_size=16):
    audio = whisperx.load_audio(video_path)
    
    result = model.transcribe(audio, batch_size=batch_size)
    
    return result["segments"]