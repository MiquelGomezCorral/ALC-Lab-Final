import os
import json
import pickle
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.meme_dataset import MemeDataset, collate_fn
from src.utils.train import train
from src.utils.train import train_transfer

# ── Constantes ────────────────────────────────────────────────────────────────
OCR_LEN      = 128
TRANS_LEN    = 128
REAS_LEN     = 256
MAX_SUBJECTS = 4
SEG_LENGTHS  = [OCR_LEN, TRANS_LEN, REAS_LEN]

DATA_DIR      = "../data/last_task/"
QWEN_EMB_PATH = "../data/EXIST 2026 Videos Dataset/training/video_embeddings_qwen3_8b-prompt.pkl"

BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VAL   = 16
NUM_WORKERS      = 4
NUM_ANNOTATORS = 10

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento multimodal — desde cero o transfer learning."
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Número de clases de la tarea (default: 2).",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="xlm-roberta-base",
        help="Nombre del encoder de texto HuggingFace (default: xlm-roberta-base).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["scratch", "transfer"],
        default="scratch",
        help="'scratch' entrena desde cero; 'transfer' carga un checkpoint previo (default: scratch).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Ruta al .pt del modelo pre-entrenado. Obligatorio si --mode=transfer.",
    )
    parser.add_argument(
        "--label_name",
        type=str,
        default="label",
        help="Nombre del campo de etiqueta en el JSON de datos (default: 'label').",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
    )

    parser.add_argument(
        "--multilabel",
        action="store_true",
        help="Si se activa, se trata la tarea como multilabel (BCE) en lugar de multiclase (KLDiv).",
    )

    parser.add_argument(
        "--annotators",
        action="store_true",
        help="El modelo utiliza anotadores.",
    )

    return parser.parse_args()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_qwen_embeddings(path):
    with open(path, "rb") as f:
        embeddings = pickle.load(f)
    emb_dim = len(next(iter(embeddings.values())))
    return embeddings, emb_dim


def main():
    args = parse_args()

    # ── Validaciones ──────────────────────────────────────────────────────────
    if args.mode == "transfer" and args.checkpoint is None:
        raise ValueError("--checkpoint es obligatorio cuando --mode=transfer.")
    if args.mode == "transfer" and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint no encontrado: {args.checkpoint}")

    print(f"🚀 Device: {device}")
    print(f"   Modo:         {args.mode}")
    print(f"   Encoder:      {args.text_encoder}")
    print(f"   Num clases:   {args.num_classes}")
    print(f"   Campo label:  {args.label_name}")
    print(f"   Balanceado:   {args.balanced}")
    if args.mode == "transfer":
        print(f"   Checkpoint:   {args.checkpoint}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)

    # ── Qwen embeddings ───────────────────────────────────────────────────────
    qwen_embeddings, qwen_emb_dim = load_qwen_embeddings(QWEN_EMB_PATH)
    print(f"Qwen embeddings cargados: {len(qwen_embeddings)} | dim: {qwen_emb_dim}")

    # ── Datos ─────────────────────────────────────────────────────────────────
    if args.label_name == "task3":
        train_data = load_json(os.path.join(DATA_DIR, "train_3.json"))
        val_data   = load_json(os.path.join(DATA_DIR, "val_3.json"))
    else:
        train_data = load_json(os.path.join(DATA_DIR, "train.json"))
        val_data   = load_json(os.path.join(DATA_DIR, "val.json"))
    print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = MemeDataset(
        train_data, tokenizer,
        qwen_embeddings=qwen_embeddings,
        ocr_len=OCR_LEN, trans_len=TRANS_LEN, reasoning_len=REAS_LEN,
        max_subjects=MAX_SUBJECTS,
        num_classes=args.num_classes,
        name_label=args.label_name,
        multilabel=args.multilabel,
        annotators=NUM_ANNOTATORS,
    )
    val_dataset = MemeDataset(
        val_data, tokenizer,
        qwen_embeddings=qwen_embeddings,
        eeg_dim=train_dataset.eeg_dim,
        et_hr_dim=train_dataset.et_hr_dim,
        ocr_len=OCR_LEN, trans_len=TRANS_LEN, reasoning_len=REAS_LEN,
        max_subjects=MAX_SUBJECTS,
        num_classes=args.num_classes,
        name_label=args.label_name,
        multilabel=args.multilabel,
        annotators=NUM_ANNOTATORS,
    )

    eeg_dim   = train_dataset.eeg_dim
    et_hr_dim = train_dataset.et_hr_dim
    print(f"EEG dim: {eeg_dim} | ET/HR dim: {et_hr_dim}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_TRAIN,
        shuffle=True, collate_fn=collate_fn,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE_VAL,
        shuffle=False, collate_fn=collate_fn,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    common_kwargs = dict(
        loader=train_loader,
        val_loader=val_loader,
        eeg_dim=eeg_dim,
        et_hr_dim=et_hr_dim,
        num_classes=args.num_classes,
        text_encoder_name=args.text_encoder,
        qwen_emb_dim=qwen_emb_dim,
        save_dir=DATA_DIR,
        seg_lengths=SEG_LENGTHS,
        phase1_epochs=5,
        phase2_epochs=20,
        es_patience=5,
        label_name=args.label_name,
        balanced=args.balanced,
        multilabel=args.multilabel,
        annotators = NUM_ANNOTATORS,
        annotations= args.annotators
    )

    if args.mode == "scratch":
        model = train(train_data=train_data, **common_kwargs)
    else:
        model = train_transfer(checkpoint_path=args.checkpoint, **common_kwargs)

    print("✅ Entrenamiento finalizado")


if __name__ == "__main__":
    main()