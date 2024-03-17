import torch
import torch.nn.functional as F
from torch import nn

def refer_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: torch.Tensor):

    loss = F.cross_entropy(inputs, targets, weight=weight)

    return loss


refer_ce_loss_jit = torch.jit.script(
    refer_ce_loss
)  # type: torch.jit.ScriptModule

class ReferringCriterion(nn.Module):
    def __init__(self, weight_dict, losses):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses

    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'masks': self.loss_masks_refer,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets)

    def loss_masks_refer(self, outputs, targets):
        src_masks = outputs["pred_masks"]
        src_minimap = outputs["pred_logits"].permute(0,2,1)
        src_nt_label = outputs["nt_label"]

        masks = [t["gt_mask_merged"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)

        target_nts = torch.stack([t["empty"] for t in targets])

        h, w = target_masks.shape[-2:]
        src_masks = F.interpolate(src_masks, (h, w), mode='bilinear', align_corners=False)

        target_minimap = F.interpolate(target_masks, (10, 10), mode='bilinear', align_corners=False).flatten(start_dim=1)

        weight = torch.FloatTensor([0.9, 1.1]).to(src_masks)

        loss_mask = \
            refer_ce_loss_jit(src_masks, target_masks.squeeze(1).long(), weight) + \
            refer_ce_loss_jit(src_minimap, target_minimap.squeeze(1).long(), weight) * 0.1 + \
            refer_ce_loss_jit(src_nt_label, target_nts, weight) * 0.1

        losses = {
            "loss_mask": loss_mask,
        }

        del src_masks
        del target_masks
        return losses

    def forward(self, outputs, targets):
        # Compute all the requested losses
        losses = {}
        losses.update(self.loss_masks_refer(outputs, targets))

        return losses


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)