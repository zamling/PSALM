import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
import sys
import os
from torch.cuda.amp import autocast

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from scipy.optimize import linear_sum_assignment
from psalm.model.mask_decoder.Mask2Former_Simplify.utils.misc import is_dist_avail_and_initialized, \
    nested_tensor_from_tensor_list
from psalm.model.mask_decoder.Mask2Former_Simplify.utils.point_features import point_sample, \
    get_uncertain_point_coords_with_randomness
from psalm.model.mask_decoder.Mask2Former_Simplify.utils.matcher import HungarianMatcher, batch_dice_loss_jit, \
    batch_sigmoid_ce_loss_jit, batch_sigmoid_focal_loss
from psalm.model.mask_decoder.Mask2Former_Simplify.utils.criterion import SetCriterion


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


class PSALM_criterion(nn.Module):

    def __init__(self, matcher, losses, num_points, oversample_ratio, importance_sample_ratio, device):
        super().__init__()
        self.matcher = matcher
        self.losses = losses
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.device = device
        self.pos_weight = torch.tensor([99.0])
        self.SEG_temp = 0.07
        self.CLASS_temp = 0.07

    def loss_labels(self, outputs, targets, indices):
        pass

    def loss_region_labels(self, outputs, targets, indices, num_masks):
        assert "pred_region_logits" in outputs
        src_logits_list = outputs['pred_region_logits']
        if src_logits_list is None:
            return {"loss_region_class": None}
        assert len(indices) == len(src_logits_list), 'batch size mismatch'
        target_query = []
        for sample_src_logits, sample_indices in zip(src_logits_list, indices):
            sample_target_query = torch.zeros_like(sample_src_logits).to(sample_src_logits.device)
            index_i, index_j = sample_indices
            sample_target_query[index_j, index_i] = 1
            target_query.append(sample_target_query)
        src_logits = torch.cat([sample_src_logits.flatten() for sample_src_logits in src_logits_list], dim=0)
        target_query = torch.cat([sample_target_query.flatten() for sample_target_query in target_query], dim=0)
        num_sample = src_logits.shape[0]
        pos_weight = (num_sample - num_masks) / num_masks
        loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(src_logits.device))
        loss_region_class = loss_func(src_logits, target_query)
        losses = {"loss_region_class": loss_region_class}
        return losses


    def loss_SEG_labels(self, outputs, targets, indices, num_masks):
        assert "pred_SEG_logits" in outputs
        # src_logits [batch_size, num_query, 1]
        pred_SEG_logits = outputs['pred_SEG_logits']
        if pred_SEG_logits is None:
            return {"loss_SEG_class": None}
        src_logits = pred_SEG_logits.float()
        src_logits = src_logits
        target_query = torch.zeros_like(src_logits).to(src_logits.device)
        for i, (index_i, _) in enumerate(indices):
            target_query[i, index_i] = 1
        num_sample = src_logits.shape[0] * src_logits.shape[1]
        pos_weight = (num_sample - num_masks) / num_masks
        loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(src_logits.device))
        # loss_func = nn.BCELoss()
        loss_SEG_class = loss_func(src_logits, target_query)
        # loss_SEG_class = -pos_weight * (target_query * torch.log(src_logits)) - neg_weight * ((1 - target_query) * torch.log(1 - src_logits))
        # loss_SEG_class = loss_SEG_class.mean()
        losses = {"loss_SEG_class": loss_SEG_class}
        return losses
    def loss_SEG_labels_focal(self, outputs, targets, indices, num_masks):
        assert "pred_SEG_logits" in outputs
        # src_logits [batch_size, num_query, 1]
        src_logits = outputs['pred_SEG_logits'].float()
        src_logits = src_logits
        target_query = torch.zeros_like(src_logits).to(src_logits.device)
        for i, (index_i, _) in enumerate(indices):
            target_query[i, index_i] = 1
        loss_SEG_class = sigmoid_focal_loss(src_logits,target_query,num_masks)
        # loss_SEG_class = -pos_weight * (target_query * torch.log(src_logits)) - neg_weight * ((1 - target_query) * torch.log(1 - src_logits))
        # loss_SEG_class = loss_SEG_class.mean()
        losses = {"loss_SEG_class": loss_SEG_class}
        return losses

    def loss_SEG_labels_concat(self, outputs, targets, indices, num_masks):
        loss_focal = self.loss_SEG_labels_focal(outputs, targets, indices, num_masks)['loss_SEG_class']
        loss_bce = self.loss_SEG_labels(outputs, targets, indices, num_masks)['loss_SEG_class']

        losses = {"loss_SEG_class": loss_focal + loss_bce * 0.5}

        return losses



    def loss_class_name_labels(self, outputs, targets, indices, num_masks):
        assert "pred_class_name_logits" in outputs
        src_logits = outputs['pred_class_name_logits']
        if src_logits == None:
            return {'loss_class_name_class': None}
        else:
            src_logits = src_logits.float()
        src_logits = src_logits

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        num_class = src_logits.shape[-1]
        target_classes = torch.full(
            src_logits.shape[:2], num_class - 1, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        weights = torch.ones(num_class)
        weights[-1] = 0.1
        loss_func = nn.CrossEntropyLoss(weight=weights.to(src_logits.device))
        loss_class_name_class = loss_func(src_logits.view(-1, num_class), target_classes.view(-1))
        losses = {'loss_class_name_class': loss_class_name_class}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
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
            'SEG_labels': self.loss_SEG_labels,
            'class_name_labels': self.loss_class_name_labels,
            'masks': self.loss_masks,
            'region_labels': self.loss_region_labels,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        # num_masks = torch.as_tensor(
        #     [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        # )
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=outputs['pred_masks'].device
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
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)







class hungarian_matcher_PSALM(HungarianMatcher):
    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_masks"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            # out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            # tgt_ids = targets[b]["labels"]
            #
            # # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # # but approximate it in 1 - proba[target class].
            # # The 1 is a constant that doesn't change the matching, it can be ommitted.
            # cost_class = -out_prob[:, tgt_ids]

            if 'pred_class_name_logits' in outputs and outputs['pred_class_name_logits'] is not None:
                class_prob = outputs['pred_class_name_logits'][b]
                class_prob = class_prob.softmax(-1)
                tgt_ids = targets[b]["labels"]
                mask_num = len(tgt_ids)
                cost_class = -class_prob[:,tgt_ids]
            else:
                cost_class = 0
                mask_num = 1

            # if 'pred_SEG_logits' in outputs and outputs['pred_SEG_logits'] is not None:
            #     seg_prob = outputs['pred_SEG_logits'][b].float() / temp
            #     seg_prob = seg_prob.sigmoid()
            #     alpha = 0.25
            #     gamma = 2.0
            #     neg_cost_class = (1 - alpha) * (seg_prob ** gamma) * (-(1 - seg_prob + 1e-8).log())
            #     pos_cost_class = alpha * ((1 - seg_prob) ** gamma) * (-(seg_prob + 1e-8).log())
            #     cost_seg = (pos_cost_class - neg_cost_class)
            #     cost_seg = cost_seg.repeat(1,mask_num)
            # else:
            #     cost_seg = 0

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask.float(),
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask.float(),
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


