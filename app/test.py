import os
import json
import pickle
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.meme_dataset import MemeDataset, collate_fn, test_collate_fn
from src.models.model import MultimodalModel
from src.models.loss import SoftLabelLoss
from src.utils.evaluate import evaluate
from src.utils.test import run_test

# ── Constantes ────────────────────────────────────────────────────────────────
OCR_LEN        = 128
TRANS_LEN      = 128
REAS_LEN       = 256
MAX_SUBJECTS   = 4
SEG_LENGTHS    = [OCR_LEN, TRANS_LEN, REAS_LEN]

TEAM_NAME = "TurboAbuelas"

DATA_DIR       = "../data/last_task/"
TEST_DIR      = f"../data/exist2026_{TEAM_NAME}/"
os.makedirs(TEST_DIR, exist_ok=True)

QWEN_EMB_PATH  = "../data/EXIST 2026 Videos Dataset/training/video_embeddings_qwen3_8b-prompt.pkl"
TEST_QWEN_EMB_PATH  = "../data/EXIST 2026 Videos Dataset/test/video_embeddings_qwen3_8b-test-prompt.pkl"

BATCH_SIZE     = 32
NUM_WORKERS    = 4
NUM_ANNOTATORS = 10

# ── Etiquetas hardcodeadas por tarea (orden = índice de clase) ─────────────────
TASK_LABELS = {
    "task1": ["NO", "YES"],
    "task2": ['NO', 'DIRECT', 'JUDGEMENTAL'],
    "task3": ['NO', 'IDEOLOGICAL-INEQUALITY', 'MISOGYNY-NON-SEXUAL-VIOLENCE', 'OBJECTIFICATION', 'SEXUAL-VIOLENCE', 'STEREOTYPING-DOMINANCE'],
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferencia/evaluación del modelo multimodal."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eval", "test"],
        default="eval",
        help="'eval' evalúa con etiquetas y métricas; 'test' genera predicciones sin etiquetas (default: eval).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Ruta al .pt del modelo a usar.",
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
        help="Nombre del campo de etiqueta / tarea: task1, task2, task3 (default: task1).",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Ruta al JSON de test/val. Si no se indica se usa val.json o test.json según el modo.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Ruta al JSON de train (para inferir dims EEG/ET). Si no se indica se usa train.json en DATA_DIR.",
    )
    # solo relevante en modo eval
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="[eval] Ruta donde guardar las predicciones JSON. Si no se indica se genera automáticamente.",
    )
    # solo relevante en modo test
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="[test] Directorio de salida para los ficheros de predicción y metadata.json. Si no se indica se usa DATA_DIR.",
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
        help="Tarea multilabel (BCE) en lugar de multiclase (KLDiv).",
    )
    parser.add_argument(
        "--annotators",
        action="store_true",
        help="El modelo utiliza anotadores.",
    )

    parser.add_argument(
        "--not_phisio",
        action="store_true",
        help="Si se activa, el modelo no utiliza ramas fisiológicas (EEG, ET/HR). Si no, solo texto.",
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_qwen_embeddings(path):
    with open(path, "rb") as f:
        embeddings = pickle.load(f)
    emb_dim = len(next(iter(embeddings.values())))
    return embeddings, emb_dim


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint no encontrado: {args.checkpoint}")

    print(f"🚀 Device:      {device}")
    print(f"   Modo:        {args.mode}")
    print(f"   Encoder:     {args.text_encoder}")
    print(f"   Num clases:  {args.num_classes}")
    print(f"   Campo label: {args.label_name}")
    print(f"   Multilabel:  {args.multilabel}")
    print(f"   Anotadores:  {args.annotators}")
    print(f"   Checkpoint:  {args.checkpoint}")
    print(f" Phisio:       {'No' if args.not_phisio else 'Sí'}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)

    # ── Qwen embeddings ───────────────────────────────────────────────────────
    qwen_embeddings, qwen_emb_dim = load_qwen_embeddings(QWEN_EMB_PATH)
    print(f"Qwen embeddings cargados: {len(qwen_embeddings)} | dim: {qwen_emb_dim}")

    if args.mode == "test":
        test_qwen_embeddings, test_qwen_emb_dim = load_qwen_embeddings(TEST_QWEN_EMB_PATH)
        print(f"Qwen embeddings cargados: {len(test_qwen_embeddings)} | dim: {test_qwen_emb_dim}")
    else:
        test_qwen_embeddings, test_qwen_emb_dim = qwen_embeddings, qwen_emb_dim

    # ── Train data (para inferir dims EEG/ET) ─────────────────────────────────
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
        training=True
    )

    eeg_dim   = train_dataset.eeg_dim
    et_hr_dim = train_dataset.et_hr_dim
    print(f"EEG dim: {eeg_dim} | ET/HR dim: {et_hr_dim}")

    # ── Dataset de inferencia ─────────────────────────────────────────────────
    if args.mode == "eval":
        if args.label_name == "task3":
            default_test = os.path.join(DATA_DIR, "val_3.json")
        else:
            default_test = os.path.join(DATA_DIR, "val.json")
    else:  # test
        if args.label_name == "task3":
            default_test = os.path.join(DATA_DIR, "test_3.json")
        else:
            default_test = os.path.join(DATA_DIR, "test.json")

    test_path = args.test_file if args.test_file else default_test
    test_data = load_json(test_path)
    print(f"{'Val' if args.mode == 'eval' else 'Test'} samples: {len(test_data)}")

    test_dataset = MemeDataset(
        test_data, tokenizer,
        qwen_embeddings=test_qwen_embeddings,
        eeg_dim=eeg_dim,
        et_hr_dim=et_hr_dim,
        ocr_len=OCR_LEN, trans_len=TRANS_LEN, reasoning_len=REAS_LEN,
        max_subjects=MAX_SUBJECTS,
        num_classes=args.num_classes,
        name_label=args.label_name,
        multilabel=args.multilabel,
        annotators=NUM_ANNOTATORS,
        training = args.mode == "eval"
    )

    # collate_fn según modo: eval necesita labels, test no
    _collate = collate_fn if args.mode == "eval" else test_collate_fn

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print(f"Batches: {len(test_loader)}")

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
        phisio= not args.not_phisio,
    ).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print("✅ Modelo cargado")

    # ── Rama eval ─────────────────────────────────────────────────────────────
    if args.mode == "eval":
        if args.multilabel:
            import torch.nn as nn
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = SoftLabelLoss(num_classes=args.num_classes)

        if args.save_path:
            save_path = args.save_path
        else:
            encoder_short = args.text_encoder.replace("/", "-")
            save_path = os.path.join(DATA_DIR, f"{encoder_short}_{args.label_name}_val.json")

        test_loss, test_auc, test_f1, test_f1_yes, ce = evaluate(
            model, criterion, test_loader, device,
            save_path=save_path,
            multilabel=args.multilabel,
        )

        print("\n" + "═" * 50)
        print("  EVAL RESULTS")
        print("═" * 50)
        print(f"  Loss   : {test_loss:.4f}")
        print(f"  AUC    : {test_auc:.4f}")
        print(f"  F1     : {test_f1:.4f}")
        print(f"  F1_yes : {test_f1_yes:.4f}")
        print(f"  CE: {ce:.4f}")
        print("═" * 50)
        print(f"  Predicciones guardadas → {save_path}")

    # ── Rama test ─────────────────────────────────────────────────────────────
    else:
        output_dir = args.output_dir if args.output_dir else TEST_DIR
        task_labels = TASK_LABELS.get(args.label_name)

        soft_path, hard_path = run_test(
            model=model,
            loader=test_loader,
            device=device,
            label_name=args.label_name,
            task_labels=task_labels,
            multilabel=args.multilabel,
            output_dir=output_dir,
            team_name=TEAM_NAME,
            args=args,
        )

        print("\n" + "═" * 50)
        print("  TEST RESULTS")
        print("═" * 50)
        print(f"  Soft → {soft_path}")
        print(f"  Hard → {hard_path}")
        print("═" * 50)


if __name__ == "__main__":
    main()