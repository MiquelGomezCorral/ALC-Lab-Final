import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftLabelLoss(nn.Module):
    """KLDiv con label smoothing opcional sobre la distribución target."""
 
    def __init__(self, smoothing: float = 0.1, num_classes: int = 2):
        super().__init__()
        self.smoothing   = smoothing
        self.num_classes = num_classes
        self.kl          = nn.KLDivLoss(reduction="batchmean")
 
    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        # Mezcla la distribución de anotadores con la uniforme
        # soft_targets_smooth = (1 - ε) * soft_targets + ε * (1/K)
        uniform = torch.full_like(soft_targets, 1.0 / self.num_classes)
        smoothed = (1 - self.smoothing) * soft_targets + self.smoothing * uniform
        return self.kl(F.log_softmax(logits, dim=-1), smoothed)