import os

from maikol_utils.file_utils import list_dir_files, load_json, save_json
from maikol_utils.file_utils import print_error
from tqdm import tqdm
import torch
import whisperx

from src.config import Configuration


def _is_cuda_oom_error(error: Exception) -> bool:
    msg = str(error).lower()
    return "out of memory" in msg and "cuda" in msg

def get_transcriptions(CONFIG: Configuration) -> str:
    """
    Placeholder function to get transcription from a video.
    In a real implementation, this would use a speech-to-text model or service.
    
    Args:
        CONFIG (Configuration): The configuration object containing paths and settings.
    Returns:
        str: The transcription of the video.
    """
    # files, n = list_dir_files(CONFIG.videos_path)#, max_files=10)
    metadata = load_json(CONFIG.videos_data)

    device = "cuda" 
    compute_type = "float32" 
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)

    transcriptions = {}
    errors = []
    for data in tqdm(metadata.values(), desc="Processing videos"):
        video_path = os.path.join(CONFIG.videos_path, data["video"])
        try:
            transcription = extract_transcription(
                model, 
                video_path, 
                language=data.get("lang", "en"), 
                batch_size=CONFIG.batch_size
            )
            transcriptions[video_path] = transcription[0]["text"] if transcription else "<no-speech>"
        except RuntimeError as e:
            if _is_cuda_oom_error(e):
                # Recover from CUDA OOM and keep the global job running.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                fallback_batch_size = max(1, CONFIG.batch_size // 2)
                try:
                    transcription = extract_transcription(
                        model,
                        video_path,
                        language=data.get("lang", "en"),
                        batch_size=fallback_batch_size,
                    )
                    transcriptions[video_path] = transcription[0]["text"] if transcription else "<no-speech>"
                    continue
                except RuntimeError as retry_error:
                    if _is_cuda_oom_error(retry_error):
                        transcriptions[video_path] = ""
                        print_error(
                            f"CUDA OOM for {video_path} (batch_size={CONFIG.batch_size}, retry={fallback_batch_size}). "
                            "Skipping video and continuing."
                        )
                        errors.append(video_path)
                        continue
                    raise
            raise
        except IndexError:
            transcriptions[video_path] = ''
            print_error(f"Error during processing for {video_path} is empty. Setting it to an empty string.")
            errors.append(video_path)

    if errors:
        print_error(f"Errors occurred for {len(errors)} videos: {errors}")

    save_json(CONFIG.transcriptions_path, transcriptions)


def extract_transcription(model, video_path, language="en", batch_size=16):
    audio = whisperx.load_audio(video_path)
    
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    
    return result["segments"]