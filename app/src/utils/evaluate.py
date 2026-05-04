import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score


def evaluate(model, criterion, loader, device, save_path=None):
    """
    Evalúa sobre `loader` y devuelve (val_loss, auc, f1_macro, f1_class1).
 
    Soporta cualquier número de clases (binario o multiclase).
 
    - Pérdida calculada sobre soft_label (consistente con training).
    - Hard label para métricas: argmax(soft_label).
    - Predicción: clase con mayor probabilidad tras softmax.
    - AUC: binario estándar si num_classes=2, OvR macro-averaged si >2.
    - f1_class1: F1 de la clase 1 en binario; F1 macro en multiclase
      (no hay una única "clase positiva" relevante en ese caso).
    - Guarda predicciones en JSON solo si f1_macro mejora el registro previo.
    """
    model.eval()
    all_probs, all_labels, total_loss = [], [], 0.0
 
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                qwen_emb       = batch["qwen_emb"].to(device),
                eeg            = batch["eeg"].to(device),
                eeg_mask       = batch["eeg_mask"].to(device),
                et_hr          = batch["et_hr"].to(device),
                et_hr_mask     = batch["et_hr_mask"].to(device),
            )
            soft_labels = batch["soft_label"].to(device)
            total_loss += criterion(logits, soft_labels).item()
 
            probs = torch.softmax(logits, dim=1).cpu().numpy()   # [B, num_classes]
            all_probs.extend(probs)
            all_labels.extend(batch["soft_label"].argmax(dim=1).numpy())
 
    all_probs  = np.array(all_probs)   # [N, num_classes]
    all_labels = np.array(all_labels)  # [N]
    preds      = all_probs.argmax(axis=1)  # clase con mayor prob
    num_classes = all_probs.shape[1]
 
    # ── AUC ──────────────────────────────────────────────────────────────
    if num_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        # OvR macro-averaged; necesita que todas las clases estén presentes
        auc = roc_auc_score(
            all_labels, all_probs,
            multi_class="ovr",
            average="macro",
        )
 
    # ── F1 ───────────────────────────────────────────────────────────────
    f1_macro = f1_score(all_labels, preds, average="macro")
 
    if num_classes == 2:
        f1_class1 = f1_score(all_labels, preds, pos_label=1, average="binary")
    else:
        # En multiclase devolvemos el F1 por clase como array informativo
        # pero el valor escalar de retorno es f1_macro (consistente con EarlyStopping)
        f1_class1 = f1_macro
 
    # ── Guardado JSON ─────────────────────────────────────────────────────
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        new_metrics = {
            "auc":      round(float(auc),       4),
            "f1_macro": round(float(f1_macro),  4),
            "f1_class1": round(float(f1_class1), 4),
        }
        save = True
        if os.path.exists(save_path):
            with open(save_path) as f:
                save = new_metrics["f1_macro"] > json.load(f).get("metrics", {}).get("f1_macro", -1)
        if save:
            print("  [evaluate] Guardando predicciones...")
            with open(save_path, "w") as f:
                json.dump({
                    "predictions": [
                        {
                            "label": int(l),
                            "pred":  int(p),
                            "probs": [round(float(pr), 4) for pr in prob_row],
                        }
                        for l, p, prob_row in zip(all_labels, preds, all_probs)
                    ],
                    "metrics": new_metrics,
                }, f, indent=2)
 
    return total_loss / len(loader), auc, f1_macro, f1_class1



class EarlyStopping:
    """
    Detiene el entrenamiento si F1-macro no mejora en `patience` épocas.
    Guarda automáticamente el mejor checkpoint.
    """

    def __init__(self, patience: int = 4, min_delta: float = 1e-4, save_path: str = "best_model.pt"):
        self.patience  = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_f1   = -1.0
        self.counter   = 0

    def step(self, f1: float, model: nn.Module) -> bool:
        """Devuelve True si hay que parar."""
        if f1 > self.best_f1 + self.min_delta:
            self.best_f1 = f1
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            return False
        self.counter += 1
        return self.counter >= self.patience