import os
import json
import pickle
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# ── Imports propios ──────────────────────────────────────────────────────────
from src.data.meme_dataset import MemeDataset, collate_fn
from src.models.model import MultimodalModel
from src.models.loss import SoftLabelLoss
from src.utils.evaluate import evaluate

# ── Constantes ────────────────────────────────────────────────────────────────
TEXT_ENCODER_NAME = "xlm-roberta-base"

OCR_LEN = 128
TRANS_LEN = 128
REAS_LEN = 256

MAX_SUBJECTS = 4
NUM_CLASSES = 2
NAME_LABEL = "task1"

SEQ_LEN = [OCR_LEN, TRANS_LEN, REAS_LEN]

DATA_DIR  = "../data/last_task/"
MODEL_PATH = "../data/last_task/xlm-roberta-base-80.6.pt"
QWEN_EMB_PATH = "../data/EXIST 2026 Videos Dataset/training/video_embeddings_qwen3_8b-prompt.pkl"

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    print(f"🚀 Device: {device}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_NAME)

    # ── Qwen embeddings ──────────────────────────────────────────────────────
    with open(QWEN_EMB_PATH, "rb") as f:
        qwen_embeddings = pickle.load(f)

    qwen_emb_dim = len(next(iter(qwen_embeddings.values())))
    print(f"Qwen emb dim: {qwen_emb_dim}")

    # ── Cargar train (para inferir dims) ─────────────────────────────────────
    train_data = load_json(os.path.join(DATA_DIR, "train_new.json"))

    train_dataset = MemeDataset(
        train_data,
        tokenizer,
        qwen_embeddings=qwen_embeddings,
        ocr_len=OCR_LEN,
        trans_len=TRANS_LEN,
        reasoning_len=REAS_LEN,
        max_subjects=MAX_SUBJECTS,
        num_classes=NUM_CLASSES,
        name_label=NAME_LABEL
    )

    eeg_dim   = train_dataset.eeg_dim
    et_hr_dim = train_dataset.et_hr_dim

    print(f"EEG dim: {eeg_dim} | ET/HR dim: {et_hr_dim}")

    # ── Cargar test ──────────────────────────────────────────────────────────
    test_data = load_json(os.path.join(DATA_DIR, "val.json"))

    test_dataset = MemeDataset(
        test_data,
        tokenizer,
        eeg_dim=eeg_dim,
        et_hr_dim=et_hr_dim,
        qwen_embeddings=qwen_embeddings,
        ocr_len=OCR_LEN,
        trans_len=TRANS_LEN,
        reasoning_len=REAS_LEN,
        max_subjects=MAX_SUBJECTS,
        num_classes=NUM_CLASSES,
        name_label=NAME_LABEL,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    print(f"Test batches: {len(test_loader)}")

    # ── Modelo ───────────────────────────────────────────────────────────────
    model = MultimodalModel(
        model_name=TEXT_ENCODER_NAME,
        eeg_dim=eeg_dim,
        et_hr_dim=et_hr_dim,
        qwen_emb_dim=qwen_emb_dim,
        text_dim=768,
        num_heads=8,
        freeze_backbone=True,
        seg_lengths=SEQ_LEN
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("✅ Modelo cargado")

    # ── Evaluación ───────────────────────────────────────────────────────────
    test_save_path = os.path.join(
        DATA_DIR, f"{TEXT_ENCODER_NAME}_test.json"
    )

    test_loss, test_auc, test_f1, test_f1_yes = evaluate(
        model,
        SoftLabelLoss(),
        test_loader,
        device,
        save_path=test_save_path
    )

    # ── Resultados ───────────────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("  TEST RESULTS")
    print("═" * 50)
    print(f"  Loss   : {test_loss:.4f}")
    print(f"  AUC    : {test_auc:.4f}")
    print(f"  F1     : {test_f1:.4f}")
    print(f"  F1_yes : {test_f1_yes:.4f}")
    print("═" * 50)
    print(f"  Predicciones guardadas → {test_save_path}")


if __name__ == "__main__":
    main()