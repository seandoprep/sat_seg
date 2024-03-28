import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

'''
Code from : https://www.kaggle.com/code/sungjunghwan/loss-function-of-image-segmentation
'''

SMOOTH = 1e-10

class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()

    def forward(self, pred_mask: Any, true_mask: Any) -> torch.Tensor:
        #flatten label and prediction tensors
        pred_mask = pred_mask.view(-1).float()
        true_mask = true_mask.view(-1).float()

        intersection = torch.sum(pred_mask * true_mask)
        total = torch.sum(pred_mask) + torch.sum(true_mask)

        # Add a small epsilon to the denominator to avoid division by zero
        dice_loss = 1.0 - (2.0 * intersection + SMOOTH) / (total + SMOOTH)
        return dice_loss
    
class DiceBCELoss(nn.Module):
    def __init__(self) -> None:
        super(DiceBCELoss, self).__init__()

    def forward(self, pred_mask: Any, true_mask: Any) -> torch.Tensor:      
        #flatten label and prediction tensors
        pred_mask = pred_mask.view(-1).float()
        true_mask = true_mask.view(-1).float()

        intersection = torch.sum(pred_mask * true_mask)
        total = torch.sum(pred_mask) + torch.sum(true_mask)

        intersection = torch.sum(pred_mask * true_mask)                            
        dice_loss = 1.0 - (2.0 * intersection + SMOOTH) / (total + SMOOTH)
        BCE = F.binary_cross_entropy(pred_mask, true_mask, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    

class IoULoss(nn.Module):
    def __init__(self) -> None:
        super(IoULoss, self).__init__()

    def forward(self, pred_mask: Any, true_mask: Any) -> torch.Tensor:
        #flatten label and prediction tensors
        pred_mask = pred_mask.view(-1).float()
        true_mask = true_mask.view(-1).float()
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = torch.sum(pred_mask * true_mask)
        total = torch.sum(pred_mask) + torch.sum(true_mask)
        union = total - intersection 
        
        IoU = (intersection + SMOOTH)/(union + SMOOTH)
                
        return 1 - IoU
    

class FocalLoss(nn.Module):
    def __init__(self) -> None:
        super(FocalLoss, self).__init__()

    def forward(self, pred_mask: Any, true_mask: Any, alpha=0.8, gamma=2):
        #flatten label and prediction tensors
        pred_mask = pred_mask.view(-1).float()
        true_mask = true_mask.view(-1).float()
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(pred_mask, true_mask, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
    

class TverskyLoss(nn.Module):
    def __init__(self) -> None:
        super(TverskyLoss, self).__init__()

    def forward(self, pred_mask: Any, true_mask: Any, alpha=0.5, beta=0.5):
        #flatten label and prediction tensors
        pred_mask = pred_mask.view(-1).float()
        true_mask = true_mask.view(-1).float()
        
        #True Positives, False Positives & False Negatives
        TP = (pred_mask * true_mask).sum()    
        FP = ((1-true_mask) * pred_mask).sum()
        FN = (true_mask * (1-pred_mask)).sum()
       
        Tversky = (TP + SMOOTH) / (TP + alpha*FP + beta*FN + SMOOTH)  
        
        return 1 - Tversky
    

class ShapeConstrainedLoss(nn.Module):
    def __init__(self) -> None:
        super(ShapeConstrainedLoss, self).__init__()
    '''
    Idea from : https://www.mdpi.com/2072-4292/14/5/1249
    '''

    def forward(self, pred_mask: Any, true_mask: Any, 
                e_pred_mask : Any, e_true_mask : Any,
                d_pred_mask : Any, d_true_mask : Any,
                alpha : float, beta : float) -> torch.Tensor:
        
        #flatten label and prediction tensors
        pred_mask = pred_mask.view(-1).float()
        true_mask = true_mask.view(-1).float()
        e_pred_mask = e_pred_mask.view(-1).float()
        e_true_mask = e_true_mask.view(-1).float()
        d_pred_mask = d_pred_mask.view(-1).float()
        d_true_mask = d_true_mask.view(-1).float()

        # Segmentation Loss
        def jaccard(pred, true):
            intersection = torch.sum(true * pred)
            union = torch.sum(true) + torch.sum(pred) - intersection
            jac = (intersection + SMOOTH) / (union + SMOOTH)
            return jac
        
        bce = F.binary_cross_entropy(pred_mask, true_mask, reduction='mean')
        jac = jaccard(pred_mask, true_mask)
        l_seg = bce + jac
        
        # Shape Loss
        l_shp = F.binary_cross_entropy(e_pred_mask, e_true_mask, reduction='mean') + jaccard(e_pred_mask, e_true_mask)
        l_rec = F.mse_loss(d_pred_mask, d_true_mask)

        # Shape Constrained Loss
        l_shape = l_seg + alpha*l_shp + beta*l_rec

        return l_shape