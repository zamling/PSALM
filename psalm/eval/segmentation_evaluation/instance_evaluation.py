# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import datetime
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.coco_evaluation import COCOEvaluator, _evaluate_predictions_on_coco
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.structures import Boxes, BoxMode, pairwise_iou, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
from iopath.common.file_io import file_lock
import shutil
from tqdm import tqdm

logger = logging.getLogger(__name__)


# modified from COCOEvaluator for instance segmetnat
class InstanceSegEvaluator(COCOEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            # all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            # num_classes = len(all_contiguous_ids)
            # assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                # assert category_id < num_classes, (
                #     f"A prediction has class={category_id}, "
                #     f"but the dataset only has {num_classes} classes and "
                #     f"predicted class id should be in [0, {num_classes - 1}]."
                # )
                assert category_id in reverse_id_mapping, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has class ids in {dataset_id_to_contiguous_id}."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    use_fast_impl=self._use_fast_impl,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res


class my_coco_evaluator(COCOEvaluator):

    def __init__(self, dataset_name, tasks=None, distributed=True, output_dir=None, *, max_dets_per_image=None,
                 use_fast_impl=True, kpt_oks_sigmas=(), allow_cached_coco=True):

        # super().__init__(dataset_name, tasks, distributed, output_dir, max_dets_per_image=max_dets_per_image,
        #                  use_fast_impl=use_fast_impl, kpt_oks_sigmas=kpt_oks_sigmas,
        #                  allow_cached_coco=allow_cached_coco)
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir

        if use_fast_impl and (COCOeval_opt is COCOeval):
            self._logger.info("Fast COCO eval is not built. Falling back to official COCO eval.")
            use_fast_impl = False
        self._use_fast_impl = use_fast_impl

        # COCOeval requires the limit on the number of detections per image (maxDets) to be a list
        # with at least 3 elements. The default maxDets in COCOeval is [1, 10, 100], in which the
        # 3rd element (100) is used as the limit on the number of detections per image when
        # evaluating AP. COCOEvaluator expects an integer for max_dets_per_image, so for COCOeval,
        # we reformat max_dets_per_image into [1, 10, max_dets_per_image], based on the defaults.
        if max_dets_per_image is None:
            max_dets_per_image = [1, 10, 100]
        else:
            max_dets_per_image = [1, 10, max_dets_per_image]
        self._max_dets_per_image = max_dets_per_image

        if tasks is not None and isinstance(tasks, CfgNode):
            kpt_oks_sigmas = (
                tasks.TEST.KEYPOINT_OKS_SIGMAS if not kpt_oks_sigmas else kpt_oks_sigmas
            )
            self._logger.warn(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            if output_dir is None:
                raise ValueError(
                    "output_dir must be provided to COCOEvaluator "
                    "for datasets not in COCO format."
                )
            self._logger.info(f"Trying to convert '{dataset_name}' to COCO format ...")

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            self.convert_to_coco_json(dataset_name, cache_path, allow_cached=allow_cached_coco)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset
        if self._do_evaluation:
            self._kpt_oks_sigmas = kpt_oks_sigmas

    def convert_to_coco_json(self, dataset_name, output_file, allow_cached=True):
        PathManager.mkdirs(os.path.dirname(output_file))
        with file_lock(output_file):
            if PathManager.exists(output_file) and allow_cached:
                logger.warning(
                    f"Using previously cached COCO format annotations at '{output_file}'. "
                    "You need to clear the cache file if your dataset has been modified."
                )
            else:
                logger.info(f"Converting annotations of dataset '{dataset_name}' to COCO format ...)")
                coco_dict = self.convert_to_coco_dict(dataset_name)

                logger.info(f"Caching COCO format annotations at '{output_file}' ...")
                tmp_file = output_file + ".tmp"
                with PathManager.open(tmp_file, "w") as f:
                    json.dump(coco_dict, f)
                shutil.move(tmp_file, output_file)

    def convert_to_coco_dict(self, dataset_name):
        """
        Convert an instance detection/segmentation or keypoint detection dataset
        in detectron2's standard format into COCO json format.

        Generic dataset description can be found here:
        https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

        COCO data format description can be found here:
        http://cocodataset.org/#format-data

        Args:
            dataset_name (str):
                name of the source dataset
                Must be registered in DatastCatalog and in detectron2's standard format.
                Must have corresponding metadata "thing_classes"
        Returns:
            coco_dict: serializable dict in COCO json format
        """

        dataset_dicts = DatasetCatalog.get(dataset_name)
        metadata = MetadataCatalog.get(dataset_name)

        # unmap the category mapping ids for COCO
        if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
            reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
        else:
            reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

        categories = [
            {"id": reverse_id_mapper(id), "name": name}
            for id, name in enumerate(metadata.thing_classes)
        ]

        logger.info("Converting dataset dicts into COCO format")
        coco_images = []
        coco_annotations = []

        for image_id, image_dict in tqdm(enumerate(dataset_dicts), total=len(dataset_dicts)):
            coco_image = {
                "id": image_dict.get("image_id", image_id),
                "width": int(image_dict["width"]),
                "height": int(image_dict["height"]),
                "file_name": str(image_dict["file_name"]),
            }
            coco_images.append(coco_image)

            anns_per_image = image_dict.get("annotations", [])
            for annotation in anns_per_image:
                # create a new dict with only COCO fields
                coco_annotation = {}

                # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
                bbox = annotation["bbox"]
                if isinstance(bbox, np.ndarray):
                    if bbox.ndim != 1:
                        raise ValueError(f"bbox has to be 1-dimensional. Got shape={bbox.shape}.")
                    bbox = bbox.tolist()
                if len(bbox) not in [4, 5]:
                    raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
                from_bbox_mode = annotation["bbox_mode"]
                to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
                bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)

                # COCO requirement: instance area
                if "segmentation" in annotation:
                    # Computing areas for instances by counting the pixels
                    segmentation = annotation["segmentation"]
                    # TODO: check segmentation type: RLE, BinaryMask or Polygon
                    if isinstance(segmentation, list):
                        polygons = PolygonMasks([segmentation])
                        area = polygons.area()[0].item()
                    elif isinstance(segmentation, dict):  # RLE
                        if isinstance(segmentation['counts'], list):
                            segmentation = mask.frPyObjects(segmentation, *segmentation['size'])
                        area = mask_util.area(segmentation).item()
                    else:
                        raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
                else:
                    # Computing areas using bounding boxes
                    if to_bbox_mode == BoxMode.XYWH_ABS:
                        bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
                        area = Boxes([bbox_xy]).area()[0].item()
                    else:
                        area = RotatedBoxes([bbox]).area()[0].item()

                if "keypoints" in annotation:
                    keypoints = annotation["keypoints"]  # list[int]
                    for idx, v in enumerate(keypoints):
                        if idx % 3 != 2:
                            # COCO's segmentation coordinates are floating points in [0, H or W],
                            # but keypoint coordinates are integers in [0, H-1 or W-1]
                            # For COCO format consistency we substract 0.5
                            # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                            keypoints[idx] = v - 0.5
                    if "num_keypoints" in annotation:
                        num_keypoints = annotation["num_keypoints"]
                    else:
                        num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

                # COCO requirement:
                #   linking annotations to images
                #   "id" field must start with 1
                coco_annotation["id"] = len(coco_annotations) + 1
                coco_annotation["image_id"] = coco_image["id"]
                coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
                coco_annotation["area"] = float(area)
                coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
                coco_annotation["category_id"] = int(reverse_id_mapper(annotation["category_id"]))

                # Add optional fields
                if "keypoints" in annotation:
                    coco_annotation["keypoints"] = keypoints
                    coco_annotation["num_keypoints"] = num_keypoints

                if "segmentation" in annotation:
                    seg = coco_annotation["segmentation"] = annotation["segmentation"]
                    if isinstance(seg, dict):  # RLE
                        if isinstance(seg['counts'], list):
                            seg = mask.frPyObjects(seg, *seg['size'])
                        counts = seg['counts']
                        if not isinstance(counts, str):
                            # make it json-serializable
                            seg["counts"] = counts.decode("ascii")

                coco_annotations.append(coco_annotation)

        logger.info(
            "Conversion finished, "
            f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}"
        )

        info = {
            "date_created": str(datetime.datetime.now()),
            "description": "Automatically generated COCO json file for Detectron2.",
        }
        coco_dict = {"info": info, "images": coco_images, "categories": categories, "licenses": None}
        if len(coco_annotations) > 0:
            coco_dict["annotations"] = coco_annotations
        return coco_dict

    def eval_single(self, img_ids=None):
        predictions = self._predictions
        self._results = OrderedDict()
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if not hasattr(self,'has_cont'):
            self.has_cont = False
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id") and not self.has_cont:
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]
        self.has_cont = True

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    cocoeval_fn=COCOeval_opt if self._use_fast_impl else COCOeval,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res
        # print(self._results['segm'])
        return copy.deepcopy(self._results)


    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(predictions, img_ids=img_ids)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, predictions, img_ids=None):
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    cocoeval_fn=COCOeval_opt if self._use_fast_impl else COCOeval,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res
