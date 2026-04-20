import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, smooth_normal=0.1):
        super().__init__()
        self.gamma        = gamma
        self.alpha        = alpha
        self.smooth_normal = smooth_normal

    def forward(self, logits, targets):
        if targets.dim() == 1:
            targets = nn.functional.one_hot(targets, 4).float()

        # Réduire la confiance sur Normal pour forcer le modèle
        # à chercher les classes rares
        targets_smoothed = targets.clone()
        targets_smoothed[:, 0] = targets[:, 0] * (1 - self.smooth_normal)

        probs = torch.sigmoid(logits)
        bce   = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets_smoothed, reduction='none')
        pt    = probs * targets_smoothed + (1-probs)*(1-targets_smoothed)
        focal = ((1 - pt) ** self.gamma) * bce

        if self.alpha is not None:
            focal = focal * self.alpha.to(logits.device)
        return focal.mean()