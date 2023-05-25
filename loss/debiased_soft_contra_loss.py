""" The Code is under Tencent Youtu Public Rule
Part of the code is adopted form SupContrast as in the comment in the class
"""
from __future__ import print_function

import torch
import torch.nn as nn

from sklearn.metrics.pairwise import cosine_similarity

class DebiasSoftConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, 
            max_probs=None, 
            labels=None, 
            mask=None, 
            pos_mask=None,
            neg_mask=None, 
            reduction="mean", 
            select_matrix=None):
        
        batch_size = features.shape[0]
        
        if mask is None and max_probs is None and labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        elif labels is None and max_probs is None:  
            mask=mask
        # mask: 偏置分数 x
        # 仅仅使用 pi*pj
        else:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            
            # pi*pj 
            if max_probs is not None and labels is not None: 
                label_mask = torch.eq(labels, labels.T).float().cuda()
                max_probs = max_probs.contiguous().view(-1, 1)
                score_mask = torch.matmul(max_probs, max_probs.T)
                label_mask = label_mask.float().cuda()
                label_mask = label_mask.mul(score_mask) 
                # pi*pj*bias_score
                if mask is not None:
                    mask=label_mask*mask
                else:
                    mask=label_mask
            # bias_score + labels
            elif max_probs is None and mask is not None and labels is not None:
                # label_mask 同类的mask mask:cosine相似度
                label_mask = torch.eq(labels, labels.T).float().cuda() 
                mask = label_mask * mask  
            # labels 
            else:
                label_mask = torch.eq(labels, labels.T).float().cuda()
                mask = label_mask

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #torch.Size([20480])= torch.Size([320, 64])
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

        # mask-out self-contrast cases 1-torch.eye
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask # mask self-self

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # compute mean of log-likelihood over positive
        
        sum_mask = mask.sum(1)
        sum_mask[sum_mask == 0] = 1
        if select_matrix is not None:
            select_matrix = select_matrix.repeat(anchor_count, contrast_count)
            mean_log_prob_pos = (mask*select_matrix * log_prob).sum(1) / sum_mask
        else:
            mean_log_prob_pos = (mask * log_prob).sum(1) / sum_mask
        
        # mean_log_prob_pos = (mask * log_prob).sum(1) / (((1-mask) * log_prob).sum(1) + 1e-9)

        # loss
        loss = -mean_log_prob_pos #torch.Size([384])
        loss = loss.view(anchor_count, batch_size) #torch.Size([2, 128])
        # loss = loss.mean()
        if reduction == "mean":
            loss = loss.mean()

        return loss

