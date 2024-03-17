# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
import sys
import os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from .point_features import point_sample, get_uncertain_point_coords_with_randomness
# from .misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list, get_world_size
from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # inputs = inputs.sigmoid()
    # inputs = inputs.flatten(1)
    # numerator = 2 * (inputs * targets).sum(-1)
    # denominator = inputs.sum(-1) + targets.sum(-1)
    # loss = 1 - (numerator + 1) / (denominator + 1)
    # return loss.sum() / num_masks
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks

sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, device):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.device = device
        empty_weight = torch.ones(self.num_classes + 1).to(device)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        if len(self.empty_weight) != self.num_classes + 1:
            self.empty_weight = torch.ones(self.num_classes + 1).to(src_logits.device)
            self.empty_weight[-1] = self.eos_coef

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight.to(src_logits.dtype))
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            data_type = src_masks.dtype
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks.float(),
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks.float(),
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks.float(),
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_binary_mask(self, target):
        y, x = target.size()
        target_onehot = torch.zeros(self.num_classes + 1, y, x)
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        return target_onehot

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             gt_masks: [bs, h_net_output, w_net_output]
        """
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        # targets = self._get_targets(gt_masks)
        # # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)
        #
        # # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_masks = sum(len(t["labels"]) for t in targets)
        # num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_masks)
        # num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        #
        # # Compute all the requested losses
        # losses = {}
        # for loss in self.losses:
        #     losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
        #
        # # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # if "aux_outputs" in outputs:
        #     for i, aux_outputs in enumerate(outputs["aux_outputs"]):
        #         indices = self.matcher(aux_outputs, targets)
        #         for loss in self.losses:
        #             l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
        #             l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
        #             losses.update(l_dict)
        #
        # return losses

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def _get_targets(self, gt_masks):
        targets = []
        for mask in gt_masks:
            binary_masks = self._get_binary_mask(mask)
            cls_label = torch.unique(mask)
            labels = cls_label[1:]
            binary_masks = binary_masks[labels]
            targets.append({'masks': binary_masks, 'labels': labels})
        return targets

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class Criterion(object):
    def __init__(self, num_classes, alpha=0.5, gamma=2, weight=None, ignore_index=0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.smooth = 1e-5
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='none')

    def get_loss(self, outputs, gt_masks):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             gt_masks: [bs, h_net_output, w_net_output]
        """
        loss_labels = 0.0
        loss_masks = 0.0
        loss_dices = 0.0
        num = gt_masks.shape[0]
        pred_logits = [outputs["pred_logits"].float()] # [bs, num_query, num_classes + 1]
        pred_masks = [outputs['pred_masks'].float()] # [bs, num_query, h, w]
        targets = self._get_targets(gt_masks, pred_logits[0].shape[1], pred_logits[0].device)
        for aux_output in outputs['aux_outputs']:
            pred_logits.append(aux_output["pred_logits"].float())
            pred_masks.append(aux_output["pred_masks"].float())

        gt_label = targets['labels'] # [bs, num_query]
        gt_mask_list = targets['masks']
        for mask_cls, pred_mask in zip(pred_logits, pred_masks):
            loss_labels += F.cross_entropy(mask_cls.transpose(1, 2), gt_label)
            # loss_masks += self.focal_loss(pred_result, gt_masks.to(pred_result.device))
            loss_dices += self.dice_loss(pred_mask, gt_mask_list)

        return loss_labels/num, loss_dices/num

    def binary_dice_loss(self, inputs, targets):
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()

    def dice_loss(self, predict, targets):
        bs = predict.shape[0]
        total_loss = 0
        for i in range(bs):
            pred_mask = predict[i]
            tgt_mask = targets[i].to(predict.device)
            dice_loss_value = self.binary_dice_loss(pred_mask, tgt_mask)
            total_loss += dice_loss_value
        return total_loss/bs

    def focal_loss(self, preds, labels):
        """
        preds: [bs, num_class + 1, h, w]
        labels: [bs, h, w]
        """
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss.mean()

    def _get_binary_mask(self, target):
        y, x = target.size()
        target_onehot = torch.zeros(self.num_classes + 1, y, x)
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        return target_onehot

    def _get_targets(self, gt_masks, num_query, device):
        binary_masks = []
        gt_labels = []
        for mask in gt_masks:
            mask_onehot = self._get_binary_mask(mask)
            cls_label = torch.unique(mask)
            labels = torch.full((num_query,), 0, dtype=torch.int64, device=gt_masks.device)
            labels[:len(cls_label)] = cls_label
            binary_masks.append(mask_onehot[cls_label])
            gt_labels.append(labels)
        return {"labels": torch.stack(gt_labels).to(device), "masks": binary_masks}