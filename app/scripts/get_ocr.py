
import os

import cv2
import whisperx
from tqdm import tqdm
from paddleocr import PaddleOCR

from maikol_utils.file_utils import print_error
from maikol_utils.file_utils import list_dir_files, load_json, save_json

from src.config import Configuration

def get_ocr_transcriptions(CONFIG: Configuration) -> str:
    """
    Placeholder function to get OCR results from a video.
    In a real implementation, this would use an OCR model or service.

    Args:
        CONFIG (Configuration): The configuration object containing paths and settings.
    Returns:
        str: The transcription of the video.
    """
    # files, n = list_dir_files(CONFIG.videos_path)#, max_files=10)
    metadata = load_json(CONFIG.videos_data)

    # Set use_gpu=True if you have an Nvidia GPU. Change lang='en' if videos are in another language.
    ocr_en = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)
    ocr_es = PaddleOCR(use_angle_cls=True, lang='es', use_gpu=True, show_log=False)

    ocr_results = {}
    errors = []
    for data in tqdm(metadata.values()[:10], desc="Processing videos"):
        video_path = os.path.join(CONFIG.videos_path, data["video"])
        transcription = extract_video_text(
            ocr_en if data.get("lang", "en") == "en" else ocr_es,
            video_path
        )
        try:
            ocr_results[video_path] = transcription[0]["text"]
        except IndexError:
            ocr_results[video_path] = ""
            # print_error(f"Transcription for {video_path} is empty. Setting it to an empty string.")
            errors.append(video_path)

    if errors:
        print_error(f"Errors occurred for {len(errors)} videos: {errors}")

    save_json(ocr_results, CONFIG.ocr_path)




def extract_video_text(ocr, video_path, frames_per_sec=1):
    cap = cv2.VideoCapture(video_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, int(fps / frames_per_sec)) 
    
    frame_count = 0
    extracted_texts = set() # Using a set automatically removes duplicate text from overlapping frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only run OCR on the specified interval (e.g., once a second)
        if frame_count % frame_interval == 0:
            result = ocr.ocr(frame, cls=True)
            
            # Parse PaddleOCR's complex output list to get just the text strings
            if result[0]:
                for line in result[0]:
                    text = line[1][0]
                    extracted_texts.add(text)
                
        frame_count += 1
        
    cap.release()

    return list(extracted_texts)