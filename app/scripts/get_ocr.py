
import os
import tempfile
import site
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import cv2
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from maikol_utils.file_utils import print_error
from maikol_utils.file_utils import load_json, save_json

from src.config import Configuration


_GOT_OCR_MODEL_NAME = "ucaslcl/GOT-OCR2_0"
_ocr_model = None
_ocr_tokenizer = None


def _normalize_text_for_match(text: str) -> str:
    """Normalize OCR output so spacing/punctuation noise does not create duplicates."""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _compact_text_for_match(text: str) -> str:
    """Aggressive normalization to compare texts with heavily split words."""
    return _normalize_text_for_match(text).replace(" ", "")


def _text_quality_score(text: str) -> tuple[float, int]:
    """Prefer outputs with fewer single-letter fragments and richer content."""
    tokens = text.split()
    if not tokens:
        return (0.0, 0)
    single_letter_tokens = sum(1 for tok in tokens if len(tok) == 1)
    fragment_penalty = single_letter_tokens / len(tokens)
    return (1.0 - fragment_penalty, len(text))


def _find_similar_text_index(compact_text: str, existing_compact_texts: list[str]) -> int | None:
    """Find an existing text that is effectively the same OCR sentence."""
    for idx, candidate in enumerate(existing_compact_texts):
        if compact_text == candidate:
            return idx

        min_len = min(len(compact_text), len(candidate))
        if min_len == 0:
            continue

        # Handle near-identical variants with minor OCR noise.
        ratio = SequenceMatcher(None, compact_text, candidate).ratio()
        if ratio >= 0.92:
            return idx

        # Handle one variant being almost-contained in the other.
        if compact_text in candidate or candidate in compact_text:
            overlap = min_len / max(len(compact_text), len(candidate))
            if overlap >= 0.90:
                return idx

    return None


def _prepare_cuda_runtime_paths() -> None:
    """Expose CUDA runtime libraries from site-packages so NVRTC can be resolved."""
    lib_dirs = []
    site_paths = set(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        site_paths.add(user_site)

    for base in sorted(site_paths):
        for rel in ("nvidia/cu13/lib", "nvidia/cuda_nvrtc/lib"):
            candidate = Path(base) / rel
            if candidate.exists():
                lib_dirs.append(str(candidate))

    if not lib_dirs:
        return

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = [p for p in current_ld.split(":") if p]
    for lib_dir in reversed(lib_dirs):
        if lib_dir not in current_parts:
            current_parts.insert(0, lib_dir)
    os.environ["LD_LIBRARY_PATH"] = ":".join(current_parts)


def _get_got_ocr():
    """Lazily load GOT-OCR2.0 once and reuse it for all videos."""
    global _ocr_model, _ocr_tokenizer

    _prepare_cuda_runtime_paths()

    if _ocr_model is not None and _ocr_tokenizer is not None:
        return _ocr_model, _ocr_tokenizer

    _ocr_tokenizer = AutoTokenizer.from_pretrained(
        _GOT_OCR_MODEL_NAME,
        trust_remote_code=True,
    )
    if _ocr_tokenizer.pad_token_id is None:
        _ocr_tokenizer.pad_token = _ocr_tokenizer.eos_token

    _ocr_model = AutoModel.from_pretrained(
        _GOT_OCR_MODEL_NAME,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cuda",
        use_safetensors=True,
        pad_token_id=_ocr_tokenizer.eos_token_id,
    ).eval()
    if hasattr(_ocr_model, "generation_config"):
        _ocr_model.generation_config.pad_token_id = _ocr_tokenizer.pad_token_id

    return _ocr_model, _ocr_tokenizer



def get_ocr_transcriptions(CONFIG: Configuration) -> str:
    """
    Extract OCR transcriptions from all videos in metadata using GOT-OCR2.0.

    Args:
        CONFIG (Configuration): The configuration object containing paths and settings.
    Returns:
        str: The transcription of the video.
    """
    metadata = load_json(CONFIG.videos_data)

    # GOT-OCR2.0 official Transformers API: load once and reuse.
    ocr_model, ocr_tokenizer = _get_got_ocr()

    ocr_results = {}
    errors = []
    for data in tqdm(list(metadata.values()), desc="Processing videos"):
        video_path = os.path.join(CONFIG.videos_path, data["video"])

        if not os.path.exists(video_path):
            ocr_results[video_path] = ""
            errors.append(video_path)
            continue

        transcription = extract_video_text(video_path, ocr_model, ocr_tokenizer)
        if transcription:
            ocr_results[video_path] = " ".join(transcription)
        else:
            ocr_results[video_path] = ""
            errors.append(video_path)

    if errors:
        print_error(f"Errors occurred for {len(errors)} videos: {errors}")

    save_json(CONFIG.ocr_path, ocr_results)


def extract_video_text(video_path, ocr_model, ocr_tokenizer, frames_per_sec=1):
    cap = cv2.VideoCapture(video_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25
    frame_interval = max(1, int(fps / frames_per_sec))
    
    frame_count = 0
    extracted_texts = []
    compact_texts = []
    quality_scores = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only run OCR on the specified interval (e.g., once a second)
        if frame_count % frame_interval == 0:
            text = _extract_text_from_frame(ocr_model, ocr_tokenizer, frame)
            cleaned = text.strip()
            if cleaned:
                compact = _compact_text_for_match(cleaned)
                similar_idx = _find_similar_text_index(compact, compact_texts)

                if similar_idx is None:
                    compact_texts.append(compact)
                    quality_scores.append(_text_quality_score(cleaned))
                    extracted_texts.append(cleaned)
                else:
                    # Keep the better rendering among similar OCR variants.
                    current_score = _text_quality_score(cleaned)
                    if current_score > quality_scores[similar_idx]:
                        quality_scores[similar_idx] = current_score
                        compact_texts[similar_idx] = compact
                        extracted_texts[similar_idx] = cleaned

        frame_count += 1

    cap.release()

    return extracted_texts


def _extract_text_from_frame(ocr_model, ocr_tokenizer, frame):
    """Extract OCR text from one frame using GOT-OCR2.0 chat() API."""
    temp_img_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_img_path = tmp.name

        if not cv2.imwrite(temp_img_path, frame):
            print_error("Failed to write temporary frame image for OCR.")
            return ""

        result = ocr_model.chat(ocr_tokenizer, temp_img_path, ocr_type="ocr")
        if isinstance(result, str):
            return result
        if result is None:
            return ""
        return str(result)
    except Exception as exc:
        print_error(f"GOT-OCR2.0 failed on current frame: {exc}")
        return ""
    finally:
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except OSError as exc:
                print_error(f"Could not remove temporary frame {temp_img_path}: {exc}")