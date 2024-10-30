import torch
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod

import random
import torch.nn as nn

from models.losses import cross_entropy
class BasePixelSampler(metaclass=ABCMeta):
    """Base class of pixel sampler."""

    def __init__(self, **kwargs):
        pass

    #@abstractmethod
    def sample(self, seg_logit, seg_label):
        """Placeholder for sample function."""


'''BasePixelSampler'''
class BasePixelSampler(object):
    def __init__(self, sample_ratio=0.5):
        self.sample_ratio = sample_ratio
    '''sample'''
    def sample(self, seg_logits, seg_targets, **kwargs):
        # seg_logits: (N, C, H, W), seg_targets: (N, H, W)
        assert seg_logits.shape[-2:] == seg_targets.shape[-2:]
        n, c, h, w = seg_logits.shape
        # num pixels
        num_pixels = h * w
        sampled_num_pixels = int(self.sample_ratio * num_pixels)
        # indices
        indices = list(range(num_pixels))
        random.shuffle(indices)
        indices = indices[:sampled_num_pixels]
        # select
        seg_logits = seg_logits.permute(2, 3, 0, 1).contiguous().reshape(h * w, n, c)
        seg_logits = seg_logits[indices].permute(1, 2, 0).contiguous().reshape(n * c, sampled_num_pixels)
        seg_targets = seg_targets.permute(1, 2, 0).contiguous().reshape(h * w, n)
        seg_targets = seg_targets[indices].permute(1, 0).contiguous().reshape(n, sampled_num_pixels)
        # return
        return seg_logits, seg_targets


'''OHEMPixelSampler'''
class OHEMPixelSampler(BasePixelSampler):
    """Online Hard Example Mining Sampler for segmentation.

    Args:
        context (nn.Module): The context of sampler, subclass of
            :obj:`BaseDecodeHead`.
        thresh (float, optional): The threshold for hard example selection.
            Below which, are prediction with low confidence. If not
            specified, the hard examples will be pixels of top ``min_kept``
            loss. Default: None.
        min_kept (int, optional): The minimum number of predictions to keep.
            Default: 100000.
    """

    def __init__(self, context, thresh=None, min_kept=100000, ignore_index=255):
        super(OHEMPixelSampler, self).__init__()
        self.context = context
        assert min_kept > 1
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index

    def sample(self, seg_logit, seg_label):
        """Sample pixels that have high loss or with low prediction confidence.

        Args:
            seg_logit (torch.Tensor): segmentation logits, shape (N, C, H, W)
            seg_label (torch.Tensor): segmentation label, shape (N, 1, H, W)

        Returns:
            torch.Tensor: segmentation weight, shape (N, H, W)
        """
        with torch.no_grad():
 #            assert seg_logit.shape[2:] == seg_label.shape[2:]
 #            assert seg_label.shape[1] == 1
 #            seg_label = seg_label.squeeze(1).long()
 #            batch_kept = self.min_kept * seg_label.size(0)
 #            #valid_mask = seg_label != self.context.ignore_index
 #            valid_mask = seg_label != self.ignore_index
 #            seg_weight = seg_logit.new_zeros(size=seg_label.size())
 #            valid_seg_weight = seg_weight[valid_mask]
 #            if self.thresh is not None:
 #                seg_prob = F.softmax(seg_logit, dim=1)
 #
 #                tmp_seg_label = seg_label.clone().unsqueeze(1)
 # #               tmp_seg_label[tmp_seg_label == self.context.ignore_index] = 0
 #                tmp_seg_label[tmp_seg_label == self.ignore_index] = 0
 #                seg_prob = seg_prob.gather(1, tmp_seg_label).squeeze(1)
 #                sort_prob, sort_indices = seg_prob[valid_mask].sort()
 #
 #                if sort_prob.numel() > 0:
 #                    min_threshold = sort_prob[min(batch_kept,
 #                                                  sort_prob.numel() - 1)]
 #                else:
 #                    min_threshold = 0.0
 #                threshold = max(min_threshold, self.thresh)
 #                valid_seg_weight[seg_prob[valid_mask] < threshold] = 1.
 #            else:
 #                if not isinstance(self.context.loss_decode, nn.ModuleList):
 #                    losses_decode = [self.context.loss_decode]
 #                else:
 #                    losses_decode = self.context.loss_decode
 #                losses = 0.0
 #                for loss_module in losses_decode:
 #                    losses += loss_module(
 #                        seg_logit,
 #                        seg_label,
 #                        weight=None,
 #                        ignore_index=self.context.ignore_index,
 #                        reduction_override='none')
 #
 #                # faster than topk according to https://github.com/pytorch/pytorch/issues/22812  # noqa
 #                _, sort_indices = losses[valid_mask].sort(descending=True)
 #                valid_seg_weight[sort_indices[:batch_kept]] = 1.
 #
 #            seg_weight[valid_mask] = valid_seg_weight
 #
 #            return seg_weight


            assert seg_logit.shape[2:] == seg_label.shape[2:]
            assert seg_label.shape[1] == 1
            seg_label = seg_label.squeeze(1).long()
            batch_kept = self.min_kept * seg_label.size(0)
            valid_mask = seg_label != self.ignore_index
            seg_weight = seg_logit.new_zeros(size=seg_label.size())
            valid_seg_weight = seg_weight[valid_mask]
            if self.thresh is not None:
                seg_prob = F.softmax(seg_logit, dim=1)

                tmp_seg_label = seg_label.clone().unsqueeze(1)
                tmp_seg_label[tmp_seg_label == self.ignore_index] = 0
                seg_prob = seg_prob.gather(1, tmp_seg_label).squeeze(1)
                sort_prob, sort_indices = seg_prob[valid_mask].sort()

                if sort_prob.numel() > 0:
                    min_threshold = sort_prob[min(batch_kept,
                                                  sort_prob.numel() - 1)]
                else:
                    min_threshold = 0.0
                threshold = max(min_threshold, self.thresh)
                valid_seg_weight[seg_prob[valid_mask] < threshold] = 1.
            else:

                losses = F.cross_entropy(seg_logit, seg_label, reduction='none', ignore_index=self.ignore_index)

                # faster than topk according to https://github.com/pytorch/pytorch/issues/22812  # noqa
                _, sort_indices = losses[valid_mask].sort(descending=True)
                valid_seg_weight[sort_indices[:batch_kept]] = 1.

            seg_weight[valid_mask] = valid_seg_weight

            return seg_weight

if __name__ == '__main__':
    ohem_sampler = OHEMPixelSampler(context='mmseg.OHEMPixelSampler', thresh=0.7, min_kept=100000)
    print(ohem_sampler)