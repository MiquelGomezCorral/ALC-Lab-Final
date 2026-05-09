import numpy as np
import torch
from torch.utils.data import Dataset


def pad_subjects(subject_list: list, max_subjects: int, feat_dim: int):
    """Lista de vectores → tensor [max_subjects, feat_dim] + máscara booleana."""
    tensor = torch.zeros(max_subjects, feat_dim)
    mask   = torch.zeros(max_subjects, dtype=torch.bool)
    for i, s in enumerate(subject_list[:max_subjects]):
        tensor[i] = torch.tensor(s, dtype=torch.float)
        mask[i]   = True
    return tensor, mask


def votes_to_soft_label(
    votes: list,
    num_classes: int,
    multilabel: bool = False,
) -> torch.Tensor:
    """
    Convierte votos de anotadores en un soft label.
 
    Modo multiclase (multilabel=False):
        Cada anotador vota UNA clase (int). El soft label es la distribución
        de frecuencias normalizada → suma 1.
        Ej: [1, 1, 0]  →  tensor([0.333, 0.667])
 
    Modo multilabel (multilabel=True):
        Cada anotador vota UNA O VARIAS clases (list[int]). El soft label
        de cada clase k es la proporción de anotadores que la marcaron →
        NO suma 1 necesariamente.
        Ej: [[5],[3],[5,3]]  →  tensor([0., 0., 0., 0.667, 0., 0.667])
 
    Args:
        votes:       Multiclase → list[int].  Multilabel → list[list[int]].
        num_classes: Número total de clases (índices 0-based).
        multilabel:  False = KLDiv mode, True = BCE mode.
 
    Returns:
        Tensor de shape (num_classes,) con valores en [0, 1].
    """
    counts = torch.zeros(num_classes)
 
    if multilabel:
        num_annotators = len(votes)
        for annotator_votes in votes:
            for cls in annotator_votes:
                counts[cls] += 1
        return counts / num_annotators          # ∈ [0,1], no suma 1
    else:
        for v in votes:
            counts[v] += 1
        return counts / counts.sum()            # distribución, suma 1


class MemeDataset(Dataset):
    """
    Campos esperados por sample:
        qwen_ocr           : str | None
        qwen_transcription : str | None
        qwen_reasoning     : str | None
        physio             : {"EEG": [...], "ET": [...], "HR": [...]}
        votes              : list[int]   (preferido: [0,1,1,0])
        label              : int         (fallback si no hay votes)
    """

    def __init__(self, data, tokenizer,qwen_embeddings: dict,
                  eeg_dim=None, et_hr_dim=None,
             ocr_len=128, trans_len=128, reasoning_len=226,
             max_subjects: int = 4, num_classes: int = 2,
             name_label: str="label", multilabel: bool = False,
             training: bool = False, annotators: int = 10):          # ← nuevos parámetros
        self.data          = data
        self.tokenizer     = tokenizer
        self.ocr_len       = ocr_len
        self.trans_len     = trans_len
        self.reasoning_len = reasoning_len
        self.qwen_embeddings  = qwen_embeddings

        self.max_subjects = max_subjects
        self.num_classes  = num_classes
        self.name_label   = name_label
        self.multilabel = multilabel

        self.annotators = annotators

        first = data[0]["physio"]
        eeg_s = first.get("EEG", [])
        et_s  = first.get("ET",  [])
        hr_s  = first.get("HR",  [])

        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id

        self.eeg_dim   = eeg_dim   or (len(eeg_s[0]) if eeg_s else 1)
        et_dim         = len(et_s[0]) if et_s else 0
        hr_dim         = len(hr_s[0]) if hr_s else 0
        self.et_hr_dim = et_hr_dim or (et_dim + hr_dim) or 1

        first_emb       = next(iter(qwen_embeddings.values()))
        self.qwen_dim   = len(first_emb)

        print(f"[MemeDataset] EEG_DIM={self.eeg_dim} | ET_HR_DIM={self.et_hr_dim} "
              f"| QWEN_DIM={self.qwen_dim} | n={len(data)}")

    def _encode_part(self, text: str | None, prefix: str, max_length: int) -> tuple:
        """Tokeniza una sola parte; devuelve (input_ids, attention_mask) como tensores 1-D."""
        content = f"{prefix} {text}" if text else ""
        enc = self.tokenizer(
            content,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
    
    def _to_onehot(self, indices, num_classes: int) -> torch.Tensor:
        vec = torch.zeros(num_classes)
        vec[indices] = 1.0
        return vec

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        id_tiktok = f"{sample['id_Tiktok']}.mp4"


        ocr_ids,   ocr_mask   = self._encode_part(sample.get("qwen_ocr"),           "[OCR]",           self.ocr_len)
        trans_ids, trans_mask = self._encode_part(sample.get("qwen_transcription"),  "[TRANSCRIPTION]", self.trans_len)
        reas_ids,  reas_mask  = self._encode_part(sample.get("qwen_reasoning"),      "[REASONING]",     self.reasoning_len)

        input_ids = torch.cat(
            [ocr_ids, trans_ids, reas_ids],
            dim=0,
        )
        attention_mask = torch.cat(
            [ocr_mask, trans_mask, reas_mask],
            dim=0,
        )

        raw_emb  = self.qwen_embeddings.get(id_tiktok)
        if raw_emb is None:
            # Fallback: vector de ceros si el meme no tiene embedding
            qwen_emb = torch.zeros(self.qwen_dim, dtype=torch.float)
        else:
            qwen_emb = torch.tensor(np.array(raw_emb), dtype=torch.float)

        # ── 2. Fisiología ─────────────────────────────────────────────────
        physio  = sample.get("physio", {})
        eeg_s   = physio.get("EEG", [])
        et_s    = physio.get("ET",  [])
        hr_s    = physio.get("HR",  [])

        n       = max(len(et_s), len(hr_s))
        et_dim  = len(et_s[0]) if et_s else 0
        hr_dim  = len(hr_s[0]) if hr_s else 0
        et_hr   = [
            (et_s[i] if i < len(et_s) else [0.0]*et_dim) +
            (hr_s[i] if i < len(hr_s) else [0.0]*hr_dim)
            for i in range(n)
        ]

        eeg_seq,   eeg_mask   = pad_subjects(eeg_s, self.max_subjects, self.eeg_dim)
        et_hr_seq, et_hr_mask = pad_subjects(et_hr, self.max_subjects, self.et_hr_dim)

        annotators = torch.zeros(self.annotators)
        if not self.multilabel:
            annotators = self._to_onehot(sample["annotators"], self.annotators)

        # ── 3. Soft label ─────────────────────────────────────────────────
        if self.training:
            if self.multilabel:
                label = self._to_onehot(sample[self.name_label], self.num_classes)
            else:
                label = votes_to_soft_label(
                votes=sample[self.name_label],
                num_classes=self.num_classes,
                multilabel=self.multilabel
            )

            return {
                "input_ids":      input_ids,
                "attention_mask": attention_mask,            
                "qwen_emb":       qwen_emb,
                "eeg":            eeg_seq,
                "eeg_mask":       eeg_mask,
                "et_hr":          et_hr_seq,
                "et_hr_mask":     et_hr_mask,
                "label":     label,
                "annotators": annotators,
            }
        else:
            return {
                    "input_ids":      input_ids,
                    "attention_mask": attention_mask,            
                    "qwen_emb":       qwen_emb,
                    "eeg":            eeg_seq,
                    "eeg_mask":       eeg_mask,
                    "et_hr":          et_hr_seq,
                    "et_hr_mask":     et_hr_mask,
                    "annotators": annotators,
                }
    
def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "qwen_emb":       torch.stack([b["qwen_emb"]       for b in batch]),
        "eeg":            torch.stack([b["eeg"]            for b in batch]),
        "eeg_mask":       torch.stack([b["eeg_mask"]       for b in batch]),
        "et_hr":          torch.stack([b["et_hr"]          for b in batch]),
        "et_hr_mask":     torch.stack([b["et_hr_mask"]     for b in batch]),
        "annotators":     torch.stack([b["annotators"]     for b in batch]),
        "label":          torch.stack([b["label"]          for b in batch])     # [B]
    }

def test_collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "qwen_emb":       torch.stack([b["qwen_emb"]       for b in batch]),
        "eeg":            torch.stack([b["eeg"]            for b in batch]),
        "eeg_mask":       torch.stack([b["eeg_mask"]       for b in batch]),
        "et_hr":          torch.stack([b["et_hr"]          for b in batch]),
        "et_hr_mask":     torch.stack([b["et_hr_mask"]     for b in batch]),
        "annotators":     torch.stack([b["annotators"]     for b in batch]),
    }