import copy
import logging

import numpy as np
import torch
import random
import cv2

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Boxes, Instances
from pycocotools import mask as coco_mask
from pycocotools.mask import encode, decode, frPyObjects


def draw_circle(mask, center, radius):
    y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
    distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    mask[distance <= radius] = 1


def enhance_with_circles(binary_mask, radius=5):
    if not isinstance(binary_mask, np.ndarray):
        binary_mask = np.array(binary_mask)

    binary_mask = binary_mask.astype(np.uint8)

    output_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    points = np.argwhere(binary_mask == 1)
    for point in points:
        draw_circle(output_mask, (point[0], point[1]), radius)
    return output_mask


def is_mask_non_empty(rle_mask):
    if rle_mask is None:
        return False
    binary_mask = decode(rle_mask)
    return binary_mask.sum() > 0


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    # if cfg.INPUT.RANDOM_FLIP != "none":
    #     augmentation.append(
    #         T.RandomFlip(
    #             horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
    #             vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
    #         )
    #     )

    augmentation.extend([
        # T.ResizeScale(
        #     min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        # ),
        T.ResizeShortestEdge(
            short_edge_length=image_size, max_size=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size), seg_pad_value=0),
    ])

    return augmentation


class COCOSemanticNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = build_transform_gen(cfg)
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def preprocess(self, dataset_dict, region_mask_type=None, mask_format='polygon',ignore_label=255):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        ignore_label = ignore_label
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format='RGB')
        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["image"] = (image - self.pixel_mean) / self.pixel_std
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        dataset_dict['transforms'] = transforms


        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict["sem_seg_file_name"]).astype("double")
        else:
            sem_seg_gt = None
        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )
        sem_seg_gt += 1
        sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        sem_seg_gt[sem_seg_gt==0] = ignore_label + 1
        sem_seg_gt -= 1
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != ignore_label]
            # check class matching:
            if 'segments_info' in dataset_dict:
                segments_info = dataset_dict["segments_info"]
                for segment_info in segments_info:
                    class_id = segment_info["category_id"]
                    if not segment_info["iscrowd"]:
                        if class_id not in classes:
                            print('Wrong samples. Can not match panoptic gt and semantic gt')
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict



def build_transform_gen_for_eval(cfg):
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    # if cfg.INPUT.RANDOM_FLIP != "none":
    #     augmentation.append(
    #         T.RandomFlip(
    #             horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
    #             vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
    #         )
    #     )

    augmentation.extend([
        T.ResizeShortestEdge(
            short_edge_length=image_size, max_size=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size), seg_pad_value=0),
    ])

    return augmentation


class COCOPanopticNewBaselineDatasetMapperForEval(COCOSemanticNewBaselineDatasetMapper):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tfm_gens = build_transform_gen_for_eval(cfg)
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
