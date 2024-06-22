import math

import torch
import torch.nn as nn
from mmcv.ops import diff_iou_rotated_2d
from mmdet.models.builder import LOSSES
from mmdet.models.losses import weighted_loss
from mmrotate.models.losses import RotatedIoULoss

from sphdet.bbox.box_formator import obb2hbb_xyxy
from sphdet.iou import fov_iou, sph2pob_standard_iou, sph_iou

from .sph2pob_transform import Sph2PobTransfrom

class KentLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(KentLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                **kwargs):
        
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * kent_loss(
            pred,
            target,
            weight,
            **kwargs)
        return loss


@weighted_loss
def kent_loss(pred, target):
    r"""Several versions of iou-based loss for OBB.

    Args:
        pred (Tensor): Predicted bboxes of format (cx, cy, w, h, a(rad)),
            shape (n, 5).
        target (Tensor): Corresponding gt bboxes, shape (n, 5).
        mode (str): Version of iou-based lossm, including "iou", "giou", 
            "diou", and "ciou". Default: 'iou'.
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    
    #_pred, _target = pred.clone(), target.clone()
    #_pred, _target = jiter_rotated_bboxes(_pred, _target)
    #ious = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0)).squeeze().clamp(min=0, max=1.0)

    #loss = 1 - ious.clamp(min=0, max=1.0)
    return loss