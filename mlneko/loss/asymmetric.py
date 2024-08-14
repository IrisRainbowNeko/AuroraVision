import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, w1=0.5, weight_file=None, clip=0.05, eps=1e-6, focal_no_grad=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.w1 = w1
        self.clip = clip
        self.focal_no_grad = focal_no_grad
        self.eps = eps

        if weight_file:
            self.cls_weight = [torch.tensor(np.load(weight_file[0])), torch.tensor(np.load(weight_file[1]))]
        else:
            self.cls_weight = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        B = x.shape[0]
        # Calculating Probabilities
        #x_sigmoid = torch.sigmoid(x)
        x_sigmoid = x
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.focal_no_grad:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            one_sided_w = one_sided_w + self.w1
            if self.focal_no_grad:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        if self.cls_weight is not None:
            self.cls_weight[0] = self.cls_weight[0].to(x.device) # pos
            self.cls_weight[1] = self.cls_weight[1].to(x.device) # neg

            cls_weight_pos = y*self.cls_weight[0].log()
            cls_weight_neg = (1-y)*self.cls_weight[1].log()
            loss = loss * (cls_weight_pos+cls_weight_neg).exp()

        return -loss.sum()/B

class AsymmetricKLLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, w1=0.5, weight_file=None, clip=0.05, eps=1e-6, focal_no_grad=False):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.w1 = w1
        self.clip = clip
        self.focal_no_grad = focal_no_grad
        self.eps = eps

        if weight_file:
            self.cls_weight = [torch.tensor(np.load(weight_file[0])), torch.tensor(np.load(weight_file[1]))]
        else:
            self.cls_weight = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        B = x.shape[0]
        # Calculating Probabilities
        #x_sigmoid = torch.sigmoid(x)
        nan_mask = torch.isnan(x)
        if nan_mask.any():
            print(torch.where(nan_mask))
        x_sigmoid = x
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = F.kl_div(xs_pos.clamp(min=self.eps).log(), y, reduction = "none")
        los_neg = F.kl_div(xs_neg.clamp(min=self.eps).log(), 1-y, reduction = "none")
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.focal_no_grad:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            one_sided_w = one_sided_w + self.w1
            if self.focal_no_grad:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        if self.cls_weight is not None:
            self.cls_weight[0] = self.cls_weight[0].to(x.device) # pos
            self.cls_weight[1] = self.cls_weight[1].to(x.device) # neg

            cls_weight_pos = y*self.cls_weight[0].log()
            cls_weight_neg = (1-y)*self.cls_weight[1].log()
            loss = loss * (cls_weight_pos+cls_weight_neg).exp()

        # 将NaN值替换为0
        nan_mask = torch.isnan(loss)
        loss[nan_mask] = 0.

        return loss.sum()/B

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss
