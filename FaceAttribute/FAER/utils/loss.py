import math
import torch
import torch.nn as nn
from loguru import logger
from torch.autograd.function import Function


class Loss(nn.Module):
    def __init__(self, mask_ratio):
        super(Loss, self).__init__()
        self.mask_ratio = mask_ratio
        logger.info(f'Using BCE and mask: {mask_ratio}')
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, target, mask=None):
        bce = torch.tensor(0)
        mask = torch.where(mask == 0, torch.ones_like(mask), self.mask_ratio * torch.ones_like(mask)) if mask is not None else torch.ones_like(logits)
        bce = self.bce(logits.clamp(min=-10, max=10), torch.where(target == 1, torch.ones_like(logits), torch.zeros_like(logits)))
        bce = torch.sum(bce * mask) / (torch.sum(mask))
        out = bce
        return out


"""
    Most borrow from: https://github.com/Alibaba-MIIL/ASL
"""


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps, max=1 - self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps, max=1 - self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

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
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                  self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                              self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                self.loss *= self.asymmetric_w
        _loss = - self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1) * 1000

        return _loss


class FocalLoss(nn.Module):
    def __init__(self,):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets):
        gpu_targets = targets.cuda()
        alpha_factor = torch.ones(gpu_targets.shape).cuda() * 0.8
        alpha_factor = torch.where(torch.eq(gpu_targets, 1), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(gpu_targets, 1), 1. - inputs, inputs)
        focal_weight = alpha_factor * torch.pow(focal_weight, 2)
        targets = targets.type(torch.FloatTensor)
        inputs = inputs.cuda()
        targets = targets.cuda()
        bce = torch.nn.functional.binary_cross_entropy(inputs, targets)
        focal_weight = focal_weight.cuda()
        cls_loss = focal_weight * bce
        return cls_loss.sum()


class LSESignLoss(nn.Module):
    def __init__(self, eta=1):
        super(LSESignLoss, self).__init__()
        self.eta = eta

    def forward(self, logits, x, w, y):
        # y is +1 -1 -1 +1 ...
        s = self.eta * logits / (x.norm(dim=1).unsqueeze(1).repeat(1, 40) * w.norm(dim=1).unsqueeze(0).repeat(logits.shape[0], 1))
        out = torch.log(1 + torch.sum(torch.exp(- s * y))) / logits.shape[0]
        return out


class LabelSmoothingCrossEntropy(torch.nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class CenterLoss(nn.Module):
    def __init__(self, feat_dim=512, num_class=7, size_average=True):
        super().__init__()
        self.feat_dim = feat_dim
        self.size_average = size_average
        self.center = nn.Parameter(torch.randn(num_class, feat_dim))
        self.centerloss = CenterLossFunction.apply
        nn.init.xavier_uniform_(self.center, gain=math.sqrt(2.0))

    def forward(self, feature, label):
        batch_size = feature.size(0)
        feature = feature.view(batch_size, -1)
        if feature.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feature.size(1)))
        means_batch = feature.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerloss(feature, label, self.center, means_batch)
        return loss


class CenterLossFunction(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(dim=0, index=label.long())
        return (feature - centers_batch).pow(2).sum()/2.0/batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers / counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None


if __name__ == '__main__':
    pass
