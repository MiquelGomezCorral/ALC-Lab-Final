import torch
import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    """
    Wrapper de XLM-RoBERTa.
    - freeze=True  : backbone congelado (Fase 1)
    - freeze=False : fine-tune completo (Fase 2)
    """

    def __init__(self, model_name: str = "xlm-roberta-base", freeze: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.freeze_backbone(freeze)

    def freeze_backbone(self, freeze: bool):
        for p in self.model.parameters():
            p.requires_grad = not freeze

    def get_sequential_output(self, input_ids, attention_mask):
        """Devuelve [B, seq_len, 768]."""
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
    
class GatedTextFusion(nn.Module):
    def __init__(self, text_dim: int = 768, seg_lengths: list = [128, 128, 226, 10, 10, 10], dropout: float = 0.1):
        super().__init__()
        self.seg_lengths = seg_lengths
        self.n_segments  = len(seg_lengths)
        self.gate_net = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim // 2, self.n_segments),
        )

    def forward(self, text_seq, gate_input=None, attention_mask=None):
        if gate_input is None:
            gate_input = text_seq.mean(dim=1)

        gates = torch.softmax(self.gate_net(gate_input), dim=-1)

        segments, start = [], 0
        for length in self.seg_lengths:
            seg = text_seq[:, start:start+length, :]   # [B, L, D]

            if attention_mask is not None:
                seg_mask = attention_mask[:, start:start+length].unsqueeze(-1).float()
                seg_mean = (seg * seg_mask).sum(dim=1) / seg_mask.sum(dim=1).clamp(min=1)
            else:
                seg_mean = seg.mean(dim=1)

            segments.append(seg_mean)
            start += length

        seg_stack = torch.stack(segments, dim=1)
        return (seg_stack * gates.unsqueeze(-1)).sum(dim=1)
    
class CrossModalAttentionBranch(nn.Module):
    def __init__(self, physio_dim: int, text_dim: int = 768,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.physio_proj = nn.Sequential(
            nn.Linear(physio_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm      = nn.LayerNorm(text_dim)
        self.importance_mlp = nn.Sequential(
            nn.Linear(text_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, physio_seq: torch.Tensor, text_seq: torch.Tensor,
                physio_mask: torch.Tensor,
                text_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        physio_seq            : [B, MAX_S, physio_dim]
        text_seq              : [B, seq_len, D]
        physio_mask           : [B, MAX_S]  True=real, False=padding
        text_key_padding_mask : [B, seq_len] True=ignorar (padding en texto)
        Returns               : [B, D]
        """
        p = self.physio_proj(physio_seq)

        attn_out, _ = self.cross_attn(
            query=p,
            key=text_seq,
            value=text_seq,
            key_padding_mask=text_key_padding_mask,   # nuevo: ignora padding del texto
        )
        p_attended = self.attn_norm(p + attn_out) * physio_mask.unsqueeze(-1).float()

        scores  = self.importance_mlp(p_attended)
        scores  = scores.masked_fill(~physio_mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=1)
        return (p_attended * weights).sum(dim=1)


class QwenGatedFusion(nn.Module):
    def __init__(self, qwen_emb_dim: int, multimodal_dim: int, text_dim: int = 768,
                 dropout: float = 0.1):
        super().__init__()
        # Qwen sube al espacio multimodal completo, no a 768
        self.proj = nn.Sequential(
            nn.Linear(qwen_emb_dim, multimodal_dim),
            nn.LayerNorm(multimodal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Gate opera en el espacio sin comprimir
        gate_in = multimodal_dim * 2
        self.gate = nn.Sequential(
            nn.Linear(gate_in, gate_in // 2),
            nn.GELU(),
            nn.Linear(gate_in // 2, multimodal_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(multimodal_dim)

    def forward(self, multimodal_raw: torch.Tensor,
                qwen_emb: torch.Tensor) -> torch.Tensor:
        """
        multimodal_raw : [B, multimodal_dim]  — sin comprimir (ej. 2304)
        qwen_emb       : [B, qwen_emb_dim]
        Returns        : [B, multimodal_dim]
        """
        qwen_proj = self.proj(qwen_emb)                                             # [B, multimodal_dim]
        gate      = self.gate(torch.cat([multimodal_raw, qwen_proj], dim=1))        # [B, multimodal_dim]
        fused     = (1 - gate) * multimodal_raw + gate * qwen_proj
        return self.norm(fused)                                                     
    
class MultimodalModel(nn.Module):
    def __init__(
        self,
        model_name:      str,
        eeg_dim:         int,
        et_hr_dim:       int,
        qwen_emb_dim:    int,
        text_dim:        int   = 768,
        num_heads:       int   = 8,
        seg_lengths:     list  = [128, 128, 226],
        freeze_backbone: bool  = True,
        dropout:         float = 0.1,
        num_classes:     int   = 2,
        num_annotators: int = 10,
        annotation: bool = False, 
        phisio: bool = True
    ):
        super().__init__()
        self.phisio = phisio

        multimodal_dim = text_dim * 3  if self.phisio else text_dim

        self.text_encoder  = TextEncoder(model_name=model_name, freeze=freeze_backbone)
        self.text_gate     = GatedTextFusion(text_dim=text_dim, seg_lengths=seg_lengths, dropout=dropout)
        self.eeg_branch    = CrossModalAttentionBranch(eeg_dim,   text_dim, num_heads, dropout)
        self.et_hr_branch  = CrossModalAttentionBranch(et_hr_dim, text_dim, num_heads, dropout)

        # Qwen fusiona directamente sobre multimodal_raw [B, 2304]
        self.qwen_fuse = QwenGatedFusion(qwen_emb_dim, multimodal_dim, text_dim, dropout)
        self.annotation = annotation

        if self.annotation:
            self.classifier = nn.Sequential(
                nn.Linear(multimodal_dim + num_annotators, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(64, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(multimodal_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(64, num_classes),
            )

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        qwen_emb:       torch.Tensor,
        eeg:            torch.Tensor,
        eeg_mask:       torch.Tensor,
        et_hr:          torch.Tensor,
        et_hr_mask:     torch.Tensor,
        annotator_ids:  torch.Tensor,
    ) -> torch.Tensor:

        text_seq   = self.text_encoder.get_sequential_output(input_ids, attention_mask)
        # gate_input = text_seq.mean(dim=1)                            # [B, D]
        gate_input = text_seq[:,0,:]

        # Mask para que CrossModal ignore padding del texto (True = ignorar)
        text_key_padding_mask = (attention_mask == 0)                # [B, seq_len]

        text_fused = self.text_gate(text_seq, gate_input, attention_mask)            # [B, D]
        eeg_ctx    = self.eeg_branch(eeg,   text_seq, eeg_mask,  text_key_padding_mask)
        et_hr_ctx  = self.et_hr_branch(et_hr, text_seq, et_hr_mask, text_key_padding_mask)

        if self.phisio:
            multimodal_raw = torch.cat([text_fused, eeg_ctx, et_hr_ctx], dim=1)  # [B, 2304]
        else:
            multimodal_raw = text_fused

        # Qwen refina en el espacio completo — devuelve [B, 2304]
        fused = self.qwen_fuse(multimodal_raw, qwen_emb)

        if self.annotation:
            final_representation = torch.cat([fused, annotator_ids], dim=1)
        else:
            final_representation = fused

        return self.classifier(final_representation)
