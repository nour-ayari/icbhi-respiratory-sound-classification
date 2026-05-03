import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for imbalanced classification.
    
    WHY: Focal loss down-weights easy examples and focuses on hard negatives,
    which is crucial for medical audio where classes are highly imbalanced.
    
    Formula: FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    where pt is the model's estimated probability for the correct class.
    
    Args:
        gamma: Focusing parameter (default=2). Higher values focus more on hard examples.
        alpha: Per-class weight (tensor of shape [num_classes]). If None, uniform weighting.
        reduction: 'mean' or 'sum'
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, num_classes] raw model outputs
            targets: [B] integer class labels OR [B, num_classes] one-hot encoded
        """
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()

        # Hard labels: [B]
        if targets.dim() == 1:
            alpha = self.alpha
            if alpha is not None:
                alpha = alpha.to(logits.device)

            ce = F.nll_loss(log_p, targets, reduction='none', weight=alpha)
            p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)

        # Soft labels (mixup/one-hot): [B, C]
        else:
            targets = targets.to(logits.device).float()
            row_sum = targets.sum(dim=1, keepdim=True).clamp_min(1e-8)
            targets = targets / row_sum

            if self.alpha is not None:
                alpha = self.alpha.to(logits.device).view(1, -1)
                ce = -(targets * alpha * log_p).sum(dim=1)
            else:
                ce = -(targets * log_p).sum(dim=1)

            p_t = (p * targets).sum(dim=1)

        # Focal loss: down-weight easy examples
        focal_weight = (1 - p_t).pow(self.gamma)
        focal_loss = focal_weight * ce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss