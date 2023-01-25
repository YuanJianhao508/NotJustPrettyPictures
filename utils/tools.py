import torch
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch import einsum
from torch.autograd import Variable

def denorm(tensor, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = torch.Tensor(std).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor(mean).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def save_image_from_tensor_batch(batch, column, path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device='cpu'):
    batch = denorm(batch, device, mean, std)
    save_image(batch, path, nrow=column)


def calculate_correct(scores, labels):
    assert scores.size(0) == labels.size(0)
    _, pred = scores.max(dim=1)
    correct = torch.sum(pred.eq(labels)).item()
    return correct


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def step_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return 0.0


def get_current_consistency_weight(epoch, weight, rampup_length, rampup_type='step'):
    if rampup_type == 'step':
        rampup_func = step_rampup
    elif rampup_type == 'linear':
        rampup_func = linear_rampup
    elif rampup_type == 'sigmoid':
        rampup_func = sigmoid_rampup
    else:
        raise ValueError("Rampup schedule not implemented")

    return weight * rampup_func(epoch, rampup_length)

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def factorization_loss(f_a, f_b):
    # empirical cross-correlation matrix
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0)+1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0)+1e-6)
    c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    loss = on_diag + 0.005 * off_diag
    # print(loss)
    return loss

### New Factorization Loss

# CLIP like loss
def clip_loss_v2(text_latents, image_latents, device):
    temperature = nn.Parameter(torch.tensor(0.07))
    text_latents, image_latents = map(lambda t: F.normalize(t, p = 2, dim = -1), (text_latents, image_latents))
    b = text_latents.size()[0]
    temp = temperature.exp()
    sim = einsum('i d, j d -> i j', text_latents, image_latents) * 1
    labels = torch.arange(b, device = device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
    # print(loss)
    return loss

# KL divergence
def KL_loss(logits_clean,logits_aug1):
    p_clean, p_aug1 = F.softmax(logits_clean, dim=1), F.softmax(logits_aug1, dim=1)
    p_mixture = torch.clamp((p_clean + p_aug1) / 3., 1e-7, 1).log()
    loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug1, reduction='batchmean')) / 2.
    return loss

# Attention Consistency
def CAM_neg(c,T=1.0):
    result = c.reshape(c.size(0), c.size(1), -1)
    result = -nn.functional.log_softmax(result / T, dim=2) / result.size(2)
    result = result.sum(2)

    return result

def CAM_pos(c,T=1.0):
    result = c.reshape(c.size(0), c.size(1), -1)
    result = nn.functional.softmax(result / T, dim=2)

    return result

def attention_consistency(c, ci_list, y, segmentation_masks=None):
    c1 = c.clone()
    c1 = Variable(c1)
    c0 = CAM_neg(c)

    # Top-k negative classes
    c1 = c1.sum(2).sum(2)
    index = torch.zeros(c1.size())
    c1[range(c0.size(0)), y] = - float("Inf")
    topk_ind = torch.topk(c1, 3, dim=1)[1]
    index[torch.tensor(range(c1.size(0))).unsqueeze(1), topk_ind] = 1
    index = index > 0.5

    # Negative CAM loss
    neg_loss = c0[index].sum() / c0.size(0)
    for ci in ci_list:
        ci = CAM_neg(ci)
        neg_loss += ci[index].sum() / ci.size(0)
    neg_loss /= len(ci_list) + 1

    # Positive CAM loss
    index = torch.zeros(c1.size())
    true_ind = [[i] for i in y]
    index[torch.tensor(range(c1.size(0))).unsqueeze(1), true_ind] = 1
    index = index > 0.5
    p0 = CAM_pos(c)[index]
    pi_list = [CAM_pos(ci)[index] for ci in ci_list]

    # Middle ground for Jensen-Shannon divergence
    p_count = 1 + len(pi_list)
    if segmentation_masks is None:
        p_mixture = p0.detach().clone()
        for pi in pi_list:
            p_mixture += pi
        p_mixture = torch.clamp(p_mixture / p_count, 1e-7, 1).log()

    else:
        mask = np.interp(segmentation_masks, (segmentation_masks.min(), segmentation_masks.max()), (0, 1))
        p_mixture = torch.from_numpy(mask).cuda()
        p_mixture = p_mixture.reshape(p_mixture.size(0), -1)
        p_mixture = torch.nn.functional.normalize(p_mixture, dim=1)

    pos_loss = nn.functional.kl_div(p_mixture, p0, reduction='batchmean')
    for pi in pi_list:
        pos_loss += nn.functional.kl_div(p_mixture, pi, reduction='batchmean')
    pos_loss /= p_count

    loss = pos_loss + neg_loss
    return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss    


###Added
def calculate_worst_acc(scores, labels, domains, acc_groups, acc_groups_count):
    assert scores.size(0) == labels.size(0)
    _, pred = scores.max(dim=1)
    correct_batch = pred.eq(labels)
    for label in [0,1]:
        for domain in [0,1]:
            mask = torch.logical_and(labels == label,domains == domain)
            n = mask.sum().item()
            corr = correct_batch[mask].sum().item()
            key = f'{domain}_{label}'
            acc_groups[key] += corr
            acc_groups_count[key] += n
    return acc_groups, acc_groups_count
