import os
import json
import numpy as np
import torch


def _next_run_id(output_dir: str, base_name: str) -> int:
    """
    Devuelve el siguiente run_id disponible para `base_name` en `output_dir`.
    Busca ficheros existentes con el patrón base_name_<N>.json y devuelve N+1.
    Si no hay ninguno, devuelve 1.
    """
    existing = []
    if os.path.exists(output_dir):
        for fname in os.listdir(output_dir):
            if fname.startswith(base_name + "_") and fname.endswith(".json"):
                suffix = fname[len(base_name) + 1 : -len(".json")]
                if suffix.isdigit():
                    existing.append(int(suffix))
    return max(existing) + 1 if existing else 1


def _build_filename(output_dir: str, label_name: str, evaluation_context: str, team_name: str) -> tuple[str, int]:
    """
    Construye la ruta completa del fichero de predicción siguiendo el formato:
      task3_<subtask>_<evaluation_context>_<team_name>_<run_id>.json

    Devuelve (filepath, run_id).
    """
    subtask   = label_name[-1]                              # "1", "2" o "3"
    base_name = f"task3_{subtask}_{evaluation_context}_{team_name}"
    run_id    = _next_run_id(output_dir, base_name)
    filename  = f"{base_name}_{run_id}.json"
    return os.path.join(output_dir, filename), run_id


def _update_metadata(output_dir: str, entry: dict) -> None:
    """
    Añade `entry` al array del fichero metadata.json en `output_dir`.
    Lo crea si no existe.
    """
    meta_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = []

    metadata.append(entry)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def run_test(
    model,
    loader,
    device: str,
    label_name: str,
    task_labels: list[str],
    multilabel: bool,
    output_dir: str,
    team_name: str,
    args,          # argparse.Namespace completo, para guardarlo en metadata
) -> tuple[str, str]:
    """
    Ejecuta inferencia sin etiquetas y genera dos ficheros de predicción:
      - soft: distribución de probabilidad por clase
      - hard: etiqueta(s) con mayor probabilidad

    Devuelve (soft_path, hard_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    all_probs = []
    all_ids   = []

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
                annotator_ids  = batch["annotators"].to(device),
            )
            if multilabel:
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_ids.extend(batch["id"])   # lista de strings tal como viene del dataset

    all_probs = np.array(all_probs)   # [N, num_classes]

    # ── Predicciones hard ─────────────────────────────────────────────────────
    if multilabel:
        # todas las clases >= 0.5; si ninguna supera el umbral, la de mayor prob
        hard_indices = []
        for prob_row in all_probs:
            selected = [i for i, p in enumerate(prob_row) if p >= 0.5]
            if not selected:
                selected = [int(np.argmax(prob_row))]
            hard_indices.append(selected)
    else:
        hard_indices = all_probs.argmax(axis=1).tolist()   # [N]

    # ── Construcción de las listas de salida ──────────────────────────────────
    soft_predictions = []
    hard_predictions = []

    for i, (sample_id, prob_row) in enumerate(zip(all_ids, all_probs)):
        # Soft: value es dict clase → probabilidad
        soft_predictions.append({
            "test_case": "EXIST2025",
            "id":        sample_id,
            "value":     {label: round(float(prob_row[j]), 4) for j, label in enumerate(task_labels)},
        })

        # Hard: value es la etiqueta (string) o lista de etiquetas en multilabel
        if multilabel:
            hard_value = [task_labels[idx] for idx in hard_indices[i]]
        else:
            hard_value = task_labels[hard_indices[i]]

        hard_predictions.append({
            "test_case": "EXIST2025",
            "id":        sample_id,
            "value":     hard_value,
        })

    # ── Guardar soft ──────────────────────────────────────────────────────────
    soft_path, soft_run_id = _build_filename(output_dir, label_name, "soft", team_name)
    with open(soft_path, "w", encoding="utf-8") as f:
        json.dump(soft_predictions, f, indent=2, ensure_ascii=False)
    print(f"  [test] Soft guardado → {soft_path}")

    # ── Guardar hard ──────────────────────────────────────────────────────────
    hard_path, hard_run_id = _build_filename(output_dir, label_name, "hard", team_name)
    with open(hard_path, "w", encoding="utf-8") as f:
        json.dump(hard_predictions, f, indent=2, ensure_ascii=False)
    print(f"  [test] Hard guardado → {hard_path}")

    # ── Actualizar metadata.json ──────────────────────────────────────────────
    args_dict = vars(args)

    for path, context, run_id in [
        (soft_path, "soft", soft_run_id),
        (hard_path, "hard", hard_run_id),
    ]:
        _update_metadata(output_dir, {
            "path":               os.path.basename(path),
            "evaluation_context": context,
            "model":              args_dict.get("checkpoint"),
            "text_encoder":       args_dict.get("text_encoder"),
            "args":               args_dict,
        })

    return soft_path, hard_path