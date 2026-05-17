import os
import torch
from tqdm import tqdm
from collections import Counter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import WeightedRandomSampler, DataLoader
import torch.nn as nn

from src.data.meme_dataset import MemeDataset, collate_fn
from src.models.model import MultimodalModel
from src.models.loss import SoftLabelLoss
from src.utils.evaluate import evaluate, EarlyStopping


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    train_data,
    loader,
    val_loader,
    eeg_dim:           int,
    et_hr_dim:         int,
    text_encoder_name: str   = "xlm-roberta-base",
    qwen_emb_dim:      int   = 4096,
    save_dir:          str   = "../data/last_task",
    phase1_epochs:     int   = 5,
    phase2_epochs:     int   = 10,
    es_patience:       int   = 4,
    seg_lengths:       list  = [128, 128, 226],
    num_classes:       int   = 2,
    label_name:        str   = "label",
    balanced:          bool  = False,
    multilabel:        bool  = False,   # ← NUEVO: activa modo multilabel
    annotators:       int   = 10,     # ← NUEVO: número de anotadores para one-hot
    annotations: bool = False,   # ← NUEVO: el modelo utiliza anotadores
    emb_model: str = "qwen",  # ← NUEVO: modelo de embeddings a usar
    phisio: bool = True,  # ← NUEVO: si False, ignora ramas fisiológicas (EEG, ET/HR)
) -> MultimodalModel:
    os.makedirs(save_dir, exist_ok=True)
    if phisio:
        json_path  = os.path.join(save_dir, f"{text_encoder_name}_{emb_model}_{label_name}.json")
        model_path = os.path.join(save_dir, f"{text_encoder_name}_{emb_model}_{label_name}.pt")
    else:
        json_path  = os.path.join(save_dir, f"{text_encoder_name}_{emb_model}_{label_name}_text_only.json")
        model_path = os.path.join(save_dir, f"{text_encoder_name}_{emb_model}_{label_name}_text_only.pt")

    model = MultimodalModel(
        model_name=text_encoder_name, eeg_dim=eeg_dim, et_hr_dim=et_hr_dim,
        qwen_emb_dim=qwen_emb_dim,
        text_dim=768, num_heads=8, freeze_backbone=True,
        seg_lengths=seg_lengths, num_classes=num_classes,
        num_annotators=annotators, annotation=annotations,
        phisio=phisio
    ).to(device)

    if multilabel:
        labels = [int(Counter(train_data[i][label_name]).most_common(1)[0][0]) for i in range(len(train_data))]
        classes, counts = torch.unique(torch.tensor(labels), return_counts=True)
        weight_per_class = 1.0 / counts.float()
        sample_weights   = weight_per_class[torch.tensor(labels)]

        # pos_weight para BCEWithLogitsLoss
        total_samples = len(labels)
        neg_counts = total_samples - counts.float()
        pos_weight = neg_counts / counts.float()  # shape: [num_classes]

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = SoftLabelLoss(num_classes=num_classes)

    # ── Balanced sampler (solo tiene sentido en multiclase) ───────────────
    if balanced and not multilabel:
        labels = [int(Counter(train_data[i][label_name]).most_common(1)[0][0]) for i in range(len(train_data))]

        classes, counts = torch.unique(torch.tensor(labels), return_counts=True)
        weight_per_class = 1.0 / counts.float()
        sample_weights   = weight_per_class[torch.tensor(labels)]
        sampler = WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(sample_weights),
            replacement = True,
        )
        balanced_loader = DataLoader(
            loader.dataset,
            batch_size  = loader.batch_size,
            sampler     = sampler,
            num_workers = loader.num_workers,
            pin_memory  = loader.pin_memory,
            collate_fn  = collate_fn,
        )
        active_loader = balanced_loader
        print(f"[Balanced] Clases: {classes.tolist()}  Counts: {counts.tolist()}")
    elif balanced and multilabel:
        # Extraer etiquetas one-hot directamente del dataset
        all_labels = [loader.dataset[i]["label"] for i in range(len(loader.dataset))]
        label_matrix = torch.stack([
            torch.tensor(l, dtype=torch.float) if not isinstance(l, torch.Tensor) else l.float()
            for l in all_labels
        ])  # (N, C)

        class_freq   = label_matrix.sum(dim=0)
        class_weight = 1.0 / (class_freq.sqrt() + 1e-6)  # en vez de class_freq directamente

        sample_weights = (label_matrix * class_weight).sum(dim=1)
        sample_weights = sample_weights / sample_weights.sum()

        sampler = WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(sample_weights),
            replacement = True,
        )
        balanced_loader = DataLoader(
            loader.dataset,
            batch_size  = loader.batch_size,
            sampler     = sampler,
            num_workers = loader.num_workers,
            pin_memory  = loader.pin_memory,
            collate_fn  = collate_fn,
        )
        active_loader = balanced_loader
        print(f"[Balanced Multilabel] Class freqs: {class_freq.tolist()}")
        print(f"[Balanced Multilabel] Sample weights min/max: {sample_weights.min():.4f} / {sample_weights.max():.4f}")
    else:
        active_loader = loader

    # ── Fase 1: backbone congelado ────────────────────────────────────────
    print("\n=== FASE 1: backbone congelado ===\n")

    params1    = [p for n, p in model.named_parameters() if "text_encoder" not in n and p.requires_grad]
    optimizer1 = torch.optim.AdamW(params1, lr=1e-5, weight_decay=0.05)
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=phase1_epochs, eta_min=1e-6)

    for epoch in range(phase1_epochs):
        model.train()
        pbar = tqdm(active_loader, desc=f"Ph1 {epoch+1}/{phase1_epochs}")
        for batch in pbar:
            optimizer1.zero_grad()
            logits = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                qwen_emb       = batch["qwen_emb"].to(device),
                eeg            = batch["eeg"].to(device),
                eeg_mask       = batch["eeg_mask"].to(device),
                et_hr          = batch["et_hr"].to(device),
                et_hr_mask     = batch["et_hr_mask"].to(device),
                annotator_ids  = batch["annotators"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer1.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        scheduler1.step()

        val_loss, auc, f1, f1_yes, ce = evaluate(
            model, criterion, val_loader, device, json_path,
            multilabel=multilabel
        )
        print(f"  → AUC={auc:.4f}  F1={f1:.4f}  F1_yes={f1_yes:.4f}  loss={val_loss:.4f}")

    # ── Fase 2: fine-tune con LR discriminativos ──────────────────────────
    print("\n=== FASE 2: fine-tune con LR discriminativos ===\n")

    model.text_encoder.freeze_backbone(False)
    enc_layers = list(model.text_encoder.model.encoder.layer)
    n          = len(enc_layers)

    param_groups = [
        {"params": model.text_encoder.model.embeddings.parameters(), "lr": 1e-6},
        *[{"params": l.parameters(), "lr": 1e-6} for l in enc_layers[:n//2]],
        *[{"params": l.parameters(), "lr": 5e-6} for l in enc_layers[n//2:]],
        {"params": [p for name, p in model.named_parameters()
                    if "text_encoder" not in name], "lr": 5e-6},
    ]
    optimizer2 = torch.optim.AdamW(param_groups, weight_decay=0.05)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=phase2_epochs, eta_min=1e-8)
    early_stop = EarlyStopping(patience=es_patience, save_path=model_path)

    best_f1 = 0.0

    for epoch in range(phase2_epochs):
        model.train()
        pbar = tqdm(active_loader, desc=f"Ph2 {epoch+1}/{phase2_epochs}")
        for batch in pbar:
            optimizer2.zero_grad()
            logits = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                qwen_emb       = batch["qwen_emb"].to(device),
                eeg            = batch["eeg"].to(device),
                eeg_mask       = batch["eeg_mask"].to(device),
                et_hr          = batch["et_hr"].to(device),
                et_hr_mask     = batch["et_hr_mask"].to(device),
                annotator_ids  = batch["annotators"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer2.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        scheduler2.step()

        val_loss, auc, f1, f1_yes, ce = evaluate(
            model, criterion, val_loader, device, json_path,
            multilabel=multilabel
        )
        marker = " ← best" if f1 > best_f1 else ""
        best_f1 = max(best_f1, f1)
        print(f"  → AUC={auc:.4f}  F1={f1:.4f}  F1_yes={f1_yes:.4f}  loss={val_loss:.4f}{marker}")

        if early_stop.step(f1, model):
            print(f"  [EarlyStopping] Sin mejora en {es_patience} épocas. Parando.")
            break

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\n✅ Mejor modelo cargado  (F1={early_stop.best_f1:.4f})")

    return model


def load_pretrained_backbone(
    model: MultimodalModel,
    checkpoint_path: str,
) -> MultimodalModel:
    """
    Carga el checkpoint del modelo original ignorando el clasificador final.
    Los pesos del clasificador se inicializan desde cero (nueva tarea).
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    filtered   = {k: v for k, v in state_dict.items() if not k.startswith("classifier.")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"  Pesos cargados del checkpoint: {checkpoint_path}")
    print(f"  Capas NO cargadas (nuevas):    {missing}")
    print(f"  Capas ignoradas (checkpoint):  {unexpected}")
    return model


def train_transfer(
    loader,
    val_loader,
    eeg_dim:           int,
    et_hr_dim:         int,
    checkpoint_path:   str,
    text_encoder_name: str   = "xlm-roberta-base",
    qwen_emb_dim:      int   = 4096,
    save_dir:          str   = "../data/transfer_task",
    phase1_epochs:     int   = 5,
    phase2_epochs:     int   = 10,
    es_patience:       int   = 4,
    seg_lengths:       list  = [128, 128, 226],
    num_classes:       int   = 2,
    label_name:        str   = "label",
    multilabel:        bool  = False,   # ← NUEVO
) -> MultimodalModel:
    """
    Entrena el modelo en una nueva tarea reutilizando los pesos pre-entrenados
    de todas las capas excepto el clasificador final.

    Fase 1 — Solo el clasificador nuevo se entrena (backbone congelado).
    Fase 2 — Fine-tune discriminativo de todo el modelo.
    """
    os.makedirs(save_dir, exist_ok=True)
    json_path  = os.path.join(save_dir, f"{text_encoder_name}_{label_name}.json")
    model_path = os.path.join(save_dir, f"{text_encoder_name}_{label_name}.pt")

    model = MultimodalModel(
        model_name=text_encoder_name,
        eeg_dim=eeg_dim,
        et_hr_dim=et_hr_dim,
        qwen_emb_dim=qwen_emb_dim,
        text_dim=768,
        num_heads=8,
        freeze_backbone=True,
        seg_lengths=seg_lengths,
        num_classes=num_classes,
    ).to(device)

    print("\n=== Cargando backbone pre-entrenado ===\n")
    model = load_pretrained_backbone(model, checkpoint_path)

    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = SoftLabelLoss(num_classes=num_classes)
    if multilabel:
        criterion = nn.BCEWithLogitsLoss()

    # ── Fase 1: solo el clasificador nuevo ───────────────────────────────
    print("\n=== FASE 1: solo clasificador nuevo (backbone congelado) ===\n")

    params1    = list(model.classifier.parameters())
    optimizer1 = torch.optim.AdamW(params1, lr=1e-3, weight_decay=0.05)
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=phase1_epochs, eta_min=1e-5)

    for epoch in range(phase1_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Ph1 {epoch+1}/{phase1_epochs}")
        for batch in pbar:
            optimizer1.zero_grad()
            logits = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                qwen_emb       = batch["qwen_emb"].to(device),
                eeg            = batch["eeg"].to(device),
                eeg_mask       = batch["eeg_mask"].to(device),
                et_hr          = batch["et_hr"].to(device),
                et_hr_mask     = batch["et_hr_mask"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer1.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        scheduler1.step()

        val_loss, auc, f1, f1_yes = evaluate(
            model, criterion, val_loader, device, json_path,
            multilabel=multilabel
        )
        print(f"  → AUC={auc:.4f}  F1={f1:.4f}  F1_yes={f1_yes:.4f}  loss={val_loss:.4f}")

    # ── Fase 2: fine-tune discriminativo completo ─────────────────────────
    print("\n=== FASE 2: fine-tune discriminativo completo ===\n")

    model.text_encoder.freeze_backbone(False)
    enc_layers = list(model.text_encoder.model.encoder.layer)
    n          = len(enc_layers)

    param_groups = [
        {"params": model.text_encoder.model.embeddings.parameters(), "lr": 1e-6},
        *[{"params": l.parameters(), "lr": 1e-6} for l in enc_layers[:n // 2]],
        *[{"params": l.parameters(), "lr": 5e-6} for l in enc_layers[n // 2:]],
        {"params": [p for name, p in model.named_parameters()
                    if "text_encoder" not in name and "classifier" not in name], "lr": 5e-6},
        {"params": model.classifier.parameters(), "lr": 5e-5},
    ]
    optimizer2 = torch.optim.AdamW(param_groups, weight_decay=0.05)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=phase2_epochs, eta_min=1e-8)
    early_stop = EarlyStopping(patience=es_patience, save_path=model_path)

    best_f1 = 0.0

    for epoch in range(phase2_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Ph2 {epoch+1}/{phase2_epochs}")
        for batch in pbar:
            optimizer2.zero_grad()
            logits = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                qwen_emb       = batch["qwen_emb"].to(device),
                eeg            = batch["eeg"].to(device),
                eeg_mask       = batch["eeg_mask"].to(device),
                et_hr          = batch["et_hr"].to(device),
                et_hr_mask     = batch["et_hr_mask"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer2.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        scheduler2.step()

        val_loss, auc, f1, f1_yes = evaluate(
            model, criterion, val_loader, device, json_path,
            multilabel=multilabel,
        )
        marker = " ← best" if f1 > best_f1 else ""
        best_f1 = max(best_f1, f1)
        print(f"  → AUC={auc:.4f}  F1={f1:.4f}  F1_yes={f1_yes:.4f}  loss={val_loss:.4f}{marker}")

        if early_stop.step(f1, model):
            print(f"  [EarlyStopping] Sin mejora en {es_patience} épocas. Parando.")
            break

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\n✅ Mejor modelo cargado  (F1={early_stop.best_f1:.4f})")

    return model