import os
import json
import pickle
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.meme_dataset import MemeDataset, collate_fn
from src.models.model import MultimodalModel
from src.models.loss import SoftLabelLoss
from src.utils.evaluate import evaluate

# ── Constantes ────────────────────────────────────────────────────────────────
OCR_LEN      = 128
TRANS_LEN    = 128
REAS_LEN     = 256
MAX_SUBJECTS = 4
SEG_LENGTHS  = [OCR_LEN, TRANS_LEN, REAS_LEN]

DATA_DIR      = "../data/last_task/"
QWEN_EMB_PATH = "../data/EXIST 2026 Videos Dataset/training/video_embeddings_qwen3_8b-prompt.pkl"

BATCH_SIZE    = 32
NUM_WORKERS   = 4
NUM_ANNOTATORS = 10

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferencia/evaluación del modelo multimodal."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Ruta al .pt del modelo a evaluar.",
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
        "--label_name",
        type=str,
        default="task1",
        help="Nombre del campo de etiqueta en el JSON de datos (default: 'task1').",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Ruta al JSON de test. Si no se indica, se usa val.json en DATA_DIR.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Ruta al JSON de train (para inferir dims EEG/ET). Si no se indica, se usa train.json en DATA_DIR.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Ruta donde guardar las predicciones JSON. Si no se indica, se genera automáticamente.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size para inferencia (default: {BATCH_SIZE}).",
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
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint no encontrado: {args.checkpoint}")

    print(f"🚀 Device: {device}")
    print(f"   Encoder:      {args.text_encoder}")
    print(f"   Num clases:   {args.num_classes}")
    print(f"   Campo label:  {args.label_name}")
    print(f"   Multilabel:   {args.multilabel}")
    print(f"   Anotadores:    {args.annotators}")
    print(f"   Checkpoint:   {args.checkpoint}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)

    # ── Qwen embeddings ───────────────────────────────────────────────────────
    qwen_embeddings, qwen_emb_dim = load_qwen_embeddings(QWEN_EMB_PATH)
    print(f"Qwen embeddings cargados: {len(qwen_embeddings)} | dim: {qwen_emb_dim}")

    # ── Datos de train (para inferir dims EEG/ET) ─────────────────────────────
    if args.label_name == "task3":
        default_train = os.path.join(DATA_DIR, "train_3.json")
    else:
        default_train = os.path.join(DATA_DIR, "train.json")

    train_path = args.train_file if args.train_file else default_train
    train_data = load_json(train_path)
    print(f"Train samples (para dims): {len(train_data)}")

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

    eeg_dim   = train_dataset.eeg_dim
    et_hr_dim = train_dataset.et_hr_dim
    print(f"EEG dim: {eeg_dim} | ET/HR dim: {et_hr_dim}")

    # ── Datos de test ─────────────────────────────────────────────────────────
    if args.label_name == "task3":
        default_test = os.path.join(DATA_DIR, "val_3.json")
    else:
        default_test = os.path.join(DATA_DIR, "val.json")

    test_path = args.test_file if args.test_file else default_test
    test_data = load_json(test_path)
    print(f"Test samples: {len(test_data)}")

    test_dataset = MemeDataset(
        test_data, tokenizer,
        qwen_embeddings=qwen_embeddings,
        eeg_dim=eeg_dim,
        et_hr_dim=et_hr_dim,
        ocr_len=OCR_LEN, trans_len=TRANS_LEN, reasoning_len=REAS_LEN,
        max_subjects=MAX_SUBJECTS,
        num_classes=args.num_classes,
        name_label=args.label_name,
        multilabel=args.multilabel,
        annotators=NUM_ANNOTATORS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print(f"Test batches: {len(test_loader)}")

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = MultimodalModel(
        model_name=args.text_encoder,
        eeg_dim=eeg_dim,
        et_hr_dim=et_hr_dim,
        qwen_emb_dim=qwen_emb_dim,
        text_dim=768,
        num_heads=8,
        freeze_backbone=True,
        seg_lengths=SEG_LENGTHS,
        num_classes=args.num_classes,
        num_annotators=NUM_ANNOTATORS,
        annotation=args.annotators,
    ).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print("✅ Modelo cargado")

    # ── Criterio ──────────────────────────────────────────────────────────────
    if args.multilabel:
        import torch.nn as nn
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = SoftLabelLoss(num_classes=args.num_classes)

    # ── Ruta de salida ────────────────────────────────────────────────────────
    if args.save_path:
        save_path = args.save_path
    else:
        encoder_short = args.text_encoder.replace("/", "-")
        save_path = os.path.join(DATA_DIR, f"{encoder_short}_{args.label_name}_test.json")

    # ── Evaluación ────────────────────────────────────────────────────────────
    test_loss, test_auc, test_f1, test_f1_yes = evaluate(
        model,
        criterion,
        test_loader,
        device,
        save_path=save_path,
        multilabel=args.multilabel,
    )

    # ── Resultados ────────────────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("  TEST RESULTS")
    print("═" * 50)
    print(f"  Loss   : {test_loss:.4f}")
    print(f"  AUC    : {test_auc:.4f}")
    print(f"  F1     : {test_f1:.4f}")
    print(f"  F1_yes : {test_f1_yes:.4f}")
    print("═" * 50)
    print(f"  Predicciones guardadas → {save_path}")


if __name__ == "__main__":
    main()