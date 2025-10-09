import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """ 
    Focal Loss for binary classification tasks.  
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        alpha: weight for the positive class (default 1.0 → balanced)
        gamma: focusing parameter (higher = focus more on hard misclassified examples)
        reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: raw logits from the model (no Sigmoid applied)
        targets: ground truth labels (0 or 1)
        """
        # Convert logits → probabilities
        probas = torch.sigmoid(inputs)
        probas = probas.clamp(min=1e-6, max=1-1e-6)  # avoid log(0)

        # Compute the focal loss
        bce_loss = F.binary_cross_entropy(probas, targets.float(), reduction='none')
        pt = torch.where(targets == 1, probas, 1 - probas)  # prob of true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss