
import os

import cv2
from tqdm import tqdm
from paddleocr import PaddleOCR

from maikol_utils.file_utils import print_error
from maikol_utils.file_utils import load_json, save_json

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
    metadata = load_json(CONFIG.videos_data)

    # PaddleOCR official API (v3.x): initialize once and reuse across all frames/videos.
    # See: https://github.com/PaddlePaddle/PaddleOCR
    ocr_en = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    ocr_es = PaddleOCR(
        lang="es",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    ocr_results = {}
    errors = []
    for data in tqdm(metadata.values(), desc="Processing videos"):
        video_path = os.path.join(CONFIG.videos_path, data["video"])

        if not os.path.exists(video_path):
            ocr_results[video_path] = ""
            errors.append(video_path)
            continue

        transcription = extract_video_text(
            ocr_en if data.get("lang", "en") == "en" else ocr_es,
            video_path
        )
        if transcription:
            ocr_results[video_path] = " ".join(transcription)
        else:
            ocr_results[video_path] = ""
            errors.append(video_path)

    if errors:
        print_error(f"Errors occurred for {len(errors)} videos: {errors}")

    save_json(ocr_results, CONFIG.ocr_path)


def extract_video_text(ocr, video_path, frames_per_sec=1):
    cap = cv2.VideoCapture(video_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25
    frame_interval = max(1, int(fps / frames_per_sec)) 
    
    frame_count = 0
    extracted_texts = []
    seen_texts = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only run OCR on the specified interval (e.g., once a second)
        if frame_count % frame_interval == 0:
            for text in _extract_text_from_frame(ocr, frame):
                cleaned = text.strip()
                if cleaned and cleaned not in seen_texts:
                    seen_texts.add(cleaned)
                    extracted_texts.append(cleaned)
                
        frame_count += 1
        
    cap.release()

    return extracted_texts


def _extract_text_from_frame(ocr, frame):
    """Extract OCR text from one frame using PaddleOCR official v3 API, with legacy fallback."""
    if hasattr(ocr, "predict"):
        results = ocr.predict(input=frame)
        texts = []
        for result in results or []:
            rec_texts = getattr(result, "rec_texts", None)
            if rec_texts is not None:
                texts.extend(rec_texts)
                continue

            if isinstance(result, dict):
                dict_texts = result.get("rec_texts")
                if dict_texts:
                    texts.extend(dict_texts)
        return texts

    # Legacy PaddleOCR API compatibility.
    legacy_result = ocr.ocr(frame)
    texts = []
    if legacy_result and legacy_result[0]:
        for line in legacy_result[0]:
            try:
                texts.append(line[1][0])
            except (IndexError, TypeError):
                continue
    return texts