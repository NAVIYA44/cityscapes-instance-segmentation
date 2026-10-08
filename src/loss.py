import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss2D(nn.Module):
    """Focal Loss for Multi-class Semantic Segmentation."""
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: (B, C, H, W), targets: (B, H, W)
        valid_mask = (targets != self.ignore_index)
        targets_clean = targets.clone()
        targets_clean[~valid_mask] = 0

        ce_loss = F.cross_entropy(logits, targets_clean, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss[valid_mask].mean() if valid_mask.sum() > 0 else ce_loss.mean()


class CenterHeatmapLoss(nn.Module):
    """MSE / Modified Focal Loss for Center Heatmap regression."""
    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred_heatmap, target_heatmap):
        # pred_heatmap: (B, 2, H, W), target_heatmap: (B, 2, H, W)
        if self.loss_type == 'mse':
            return F.mse_loss(pred_heatmap, target_heatmap, reduction='mean')
        elif self.loss_type == 'focal':
            # Modified Focal loss for keypoint/center detection (CornerNet / Panoptic-DeepLab style)
            pos_mask = (target_heatmap == 1.0)
            neg_mask = (target_heatmap < 1.0)

            pos_loss = torch.log(pred_heatmap + 1e-6) * torch.pow(1 - pred_heatmap, 2) * pos_mask
            neg_loss = torch.log(1 - pred_heatmap + 1e-6) * torch.pow(pred_heatmap, 2) * torch.pow(1 - target_heatmap, 4) * neg_mask

            num_pos = pos_mask.float().sum()
            loss = -(pos_loss.sum() + neg_loss.sum())
            return loss / torch.clamp(num_pos, min=1.0)
        else:
            raise ValueError(f"Unknown center loss type: {self.loss_type}")


class CenterOffsetLoss(nn.Module):
    """Masked L1 / Smooth L1 Loss for 2D spatial center offset regression."""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred_offset, target_offset, offset_weight):
        # pred_offset: (B, 2, H, W), target_offset: (B, 2, H, W), offset_weight: (B, 1, H, W)
        diff = F.smooth_l1_loss(pred_offset, target_offset, beta=self.beta, reduction='none')
        masked_diff = diff * offset_weight
        num_valid = offset_weight.sum() * 2.0
        return masked_diff.sum() / torch.clamp(num_valid, min=1.0)


class ProposalFreeInstanceLoss(nn.Module):
    """
    Combined Loss for Proposal-Free Dense Instance Segmentation.
    
    L_total = lambda_sem * L_sem + lambda_center * L_center + lambda_offset * L_offset
    """
    def __init__(self, lambda_sem=1.0, lambda_center=200.0, lambda_offset=0.01, ignore_index=255):
        super().__init__()
        self.lambda_sem = lambda_sem
        self.lambda_center = lambda_center
        self.lambda_offset = lambda_offset

        self.sem_loss_fn = FocalLoss2D(ignore_index=ignore_index)
        self.center_loss_fn = CenterHeatmapLoss(loss_type='mse')
        self.offset_loss_fn = CenterOffsetLoss()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict containing 'semantic_logits', 'center_heatmap', 'offset_map'
            targets: dict containing 'semantic_target', 'center_target', 'offset_target', 'offset_weight'
        """
        sem_loss = self.sem_loss_fn(predictions['semantic_logits'], targets['semantic_target'])
        center_loss = self.center_loss_fn(predictions['center_heatmap'], targets['center_target'])
        offset_loss = self.offset_loss_fn(predictions['offset_map'], targets['offset_target'], targets['offset_weight'])

        total_loss = (self.lambda_sem * sem_loss + 
                      self.lambda_center * center_loss + 
                      self.lambda_offset * offset_loss)

        return {
            'total_loss': total_loss,
            'sem_loss': sem_loss,
            'center_loss': center_loss,
            'offset_loss': offset_loss
        }
