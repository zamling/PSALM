import argparse
import random

import torch
import os
import json
from tqdm import tqdm
import shortuuid
from pycocotools import mask
import numpy as np
import cv2
from psalm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX, CLS_TOKEN_INDEX
from psalm.model.builder import load_pretrained_model
from psalm.utils import disable_torch_init
from psalm.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from psalm.eval.segmentation_evaluation.panoptic_evaluation import my_coco_panoptic_evaluator, my_SemSegEvaluator
from transformers import StoppingCriteria, StoppingCriteriaList

from torch.utils.data import Dataset, DataLoader

from psalm import conversation as conversation_lib
from detectron2.data.datasets import load_sem_seg
from PIL import Image
import math
import copy
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from psalm.train.train_datasets import COCO_semantic_dataset
import transformers
from segmentation_evaluation import openseg_classes

PASCAL_CTX_459_CATEGORIES=openseg_classes.get_pascal_ctx_459_categories_with_prompt_eng()

PASCAL_CTX_459_COLORS = [k["color"] for k in PASCAL_CTX_459_CATEGORIES]
PASCAL_CTX_59_CATEGORIES=openseg_classes.get_pascal_ctx_59_categories_with_prompt_eng()

PASCAL_CTX_59_COLORS = [k["color"] for k in PASCAL_CTX_59_CATEGORIES]
PASCAL_VOC_20_CATEGORIES = openseg_classes.get_pascal_21_categories_with_prompt_eng()[1:] # remove background

PASCAL_VOC_20_COLORS = [k["color"] for k in PASCAL_VOC_20_CATEGORIES]
ADE20K_150_CATEGORIES = openseg_classes.get_ade20k_categories_with_prompt_eng()

ADE20k_COLORS = [k["color"] for k in ADE20K_150_CATEGORIES]
@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    model_path: Optional[str] = field(default="/path/to/model")
    mask_config: Optional[str] = field(default="./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    model_map_name: str = 'psalm'
    version: str = 'llava_phi'
    output_dir: str = './output/panoptic_segmentation'
    segmentation: bool = True
    eval_batch_size: int = 1
    dataloader_num_workers: int = 4
    seg_task: Optional[str] = field(default="semantic")
    ov_task_list: Optional[str] = field(default="ctx_59")




@dataclass
class DataCollatorForCOCODatasetV2(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        for instance in instances:
            for key in ['input_ids', 'labels', 'image']:
                del instance[key]
        batch['seg_info'] = [instance for instance in instances]

        if 'dataset_type' in instances[0]:
            batch['dataset_type'] = [instance['dataset_type'] for instance in instances]

        if 'class_name_ids' in instances[0]:
            class_name_ids = [instance['class_name_ids'] for instance in instances]
            if any(x.shape != class_name_ids[0].shape for x in class_name_ids):
                batch['class_name_ids'] = torch.nn.utils.rnn.pad_sequence(
                    class_name_ids,
                    batch_first=True,
                    padding_value=-1,
                )
            else:
                batch['class_name_ids'] = torch.stack(class_name_ids, dim=0)
        if 'token_refer_id' in instances[0]:
            token_refer_id = [instance['token_refer_id'] for instance in instances]
            batch['token_refer_id'] = token_refer_id
        if 'cls_indices' in instances[0]:
            cls_indices = [instance['cls_indices'] for instance in instances]
            if any(x.shape != cls_indices[0].shape for x in cls_indices):
                batch['cls_indices'] = torch.nn.utils.rnn.pad_sequence(
                    cls_indices,
                    batch_first=True,
                    padding_value=-1,
                )
            else:
                batch['cls_indices'] = torch.stack(cls_indices, dim=0)
        if 'random_idx' in instances[0]:
            random_idxs = [instance['random_idx'] for instance in instances]
            batch['random_idx'] = torch.stack(random_idxs, dim=0)
        if 'class_name_embedding_indices' in instances[0]:
            class_name_embedding_indices = [instance['class_name_embedding_indices'] for instance in instances]
            class_name_embedding_indices = torch.nn.utils.rnn.pad_sequence(
                class_name_embedding_indices,
                batch_first=True,
                padding_value=0)
            batch['class_name_embedding_indices'] = class_name_embedding_indices
        if 'refer_embedding_indices' in instances[0]:
            refer_embedding_indices = [instance['refer_embedding_indices'] for instance in instances]
            refer_embedding_indices = torch.nn.utils.rnn.pad_sequence(
                refer_embedding_indices,
                batch_first=True,
                padding_value=0)
            batch['refer_embedding_indices'] = refer_embedding_indices
        if 'class_id_mapping' in instances[0]:
            class_id_mapping = [instance['class_id_mapping'] for instance in instances]
            batch['class_id_mapping'] = class_id_mapping

        return batch

def _get_ctx459_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in PASCAL_CTX_459_CATEGORIES]
    assert len(stuff_ids) == 459, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in PASCAL_CTX_459_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret

def _get_ctx59_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in PASCAL_CTX_59_CATEGORIES]
    assert len(stuff_ids) == 59, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in PASCAL_CTX_59_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret

def _get_pascal20_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in PASCAL_VOC_20_CATEGORIES]
    assert len(stuff_ids) == 20, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in PASCAL_VOC_20_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret

def get_ade150_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in ADE20K_150_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in ADE20K_150_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in ADE20K_150_CATEGORIES]
    stuff_colors = [k["color"] for k in ADE20K_150_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(ADE20K_150_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta

OV_SEM_DICT={
    'ade_150':
        {
            'json_path': '/home/hk/yyma/data/ov_sem_data/ADEChallengeData2016',
            'image_path': 'images/validation',
            'gt_path': 'annotations_detectron2/validation',
            'ignore_label': 255,
            'tot_cls': 150,
            'gt_ext': "png",
            'image_ext': "jpg",
            'get_mete_method': get_ade150_metadata
        },
    'ctx_459':
        {
            'json_path': '/home/hk/yyma/data/ov_sem_data/pascal_ctx_d2',
            'image_path': 'images/validation',
            'gt_path': 'annotations_ctx459/validation',
            'ignore_label': 65535,
            'tot_cls':459,
            'gt_ext':"tif",
            'image_ext':"jpg",
            'get_mete_method':_get_ctx459_meta
        },
    'ctx_59':
        {
            'json_path': '/home/hk/yyma/data/ov_sem_data/pascal_ctx_d2',
            'image_path': 'images/validation',
            'gt_path': 'annotations_ctx59/validation',
            'ignore_label': 255,
            'tot_cls': 59,
            'gt_ext': "png",
            'image_ext': "jpg",
            'get_mete_method': _get_ctx59_meta
        },
    'pc_20':
        {
            'json_path': '/home/hk/yyma/data/ov_sem_data/pascal_voc_d2',
            'image_path': 'images/validation',
            'gt_path': 'annotations_pascal20/validation',
            'ignore_label': 255,
            'tot_cls': 20,
            'gt_ext': "png",
            'image_ext': "jpg",
            'get_mete_method': _get_pascal20_meta
        }
}


class common_semantic_dataset(COCO_semantic_dataset):
    def __init__(self, task_name, tokenizer, data_args, is_train=True):
        super(common_semantic_dataset).__init__()
        task_info = OV_SEM_DICT[task_name]
        self.semantic_image_path = os.path.join(task_info['json_path'],task_info['image_path'])
        self.semantic_gt_path = os.path.join(task_info['json_path'],task_info['gt_path'])
        self.cate = task_info['get_mete_method']()
        self.data = load_sem_seg(gt_root=self.semantic_gt_path, image_root=self.semantic_image_path, gt_ext=task_info["gt_ext"],
                                 image_ext=task_info["image_ext"])

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.mask_format = 'polygon'
        self.common_id_to_cont_id = self.cate['stuff_dataset_id_to_contiguous_id'] if 'stuff_dataset_id_to_contiguous_id' in self.cate else None
        self.common_class_name = self.cate['stuff_classes']
        self.common_class_id = list(range(len(self.common_class_name)))
        self.ignore_label = task_info['ignore_label']
        self.total_class = task_info['tot_cls']

    def preprocess_class_name(self, CLS_token='[SEG]', current_sample_class_name=None):
        tokenized = [self.tokenizer.encode(class_name, add_special_tokens=False) for class_name in
                     current_sample_class_name]
        tokenized_class_names = [tokens + [self.tokenizer.encode(CLS_token, add_special_tokens=False)[0]] for tokens in
                                 tokenized]
        class_name_id = [token for sublist in tokenized_class_names for token in sublist]
        class_name_id = torch.tensor(class_name_id)
        cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
        cls_indices = torch.tensor(cls_indices)

        return class_name_id, cls_indices
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = self.data[idx]

        data_dict = data

        if isinstance(self.data_args.image_processor, dict):
            processor = self.data_args.image_processor['semantic']
        else:
            processor = self.data_args.image_processor
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format,ignore_label=self.ignore_label)
        # instruction = data['instruction']
        instruction = 'Panoptic Segmentation: You need to segment all objects '
        prefix_inst = 'This is an image <image>, Please do Panoptic Segmentation.'



        num_class = self.total_class
        full2sample_mapping = {}
        if len(self.common_class_id) > num_class:
            current_sample_class_id = data_dict['instances'].gt_classes.numpy().tolist()
            num_negatives = num_class - 1 - len(current_sample_class_id)
            potential_negative_ids = list(set(self.common_class_id) - set(current_sample_class_id))
            negative_sample_ids = np.random.choice(potential_negative_ids, num_negatives, replace=False)
            pick_class_id = current_sample_class_id + list(negative_sample_ids)
        else:
            pick_class_id = self.common_class_id
        # random.shuffle(pick_class_id)
        for new_id, original_id in enumerate(pick_class_id):
            full2sample_mapping[original_id] = new_id
        if len(pick_class_id) > 200:
            current_sample_class_name = [self.common_class_name[id].split(',')[0] for id in
                                         pick_class_id] + ['background']
        else:
            current_sample_class_name = [self.common_class_name[id] for id in
                                         pick_class_id] + ['background']






        category = '<cls>, ' * (len(current_sample_class_name) - 1) + '<cls>.'

        sources_value = f'\nThis is all the candidate categories: {category}\n'

        sources = [[{'from': 'human', 'value':  prefix_inst + sources_value},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]

        # sources = self.preprocess_multimodal(copy.deepcopy(sources))

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        class_name_ids, cls_indices = self.preprocess_class_name(current_sample_class_name=current_sample_class_name)
        class_name_embedding_indices = torch.zeros_like(input_ids)
        class_name_embedding_indices[input_ids == CLS_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]

        data_dict['class_name_ids'] = class_name_ids
        data_dict['cls_indices'] = cls_indices
        data_dict['class_name_embedding_indices'] = class_name_embedding_indices
        data_dict['class_id_mapping'] = {value: key for key, value in full2sample_mapping.items()}
        return data_dict

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if stop == last_token:
                return True
        return False


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def evaluation(data_args,ov_task=None):
    disable_torch_init()
    model_path = os.path.expanduser(data_args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,mask_config=data_args.mask_config,model_args=data_args)
    data_args.image_processor = image_processor
    data_args.is_multimodal = True

    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]
    eval_dataset = common_semantic_dataset(task_name=ov_task, tokenizer=tokenizer,
                                           data_args=data_args)
    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer)
    dataloader_params = {
        "batch_size": data_args.eval_batch_size,
        "num_workers": data_args.dataloader_num_workers,
    }
    eval_dataloader = DataLoader(eval_dataset, batch_size=dataloader_params['batch_size'], collate_fn=data_collator,
                                 num_workers=dataloader_params['num_workers'])

    def load_instruction_dataset():
        # return COCO_instruction_dataset(json_path=data_args.json_path, tokenizer=tokenizer, data_args=data_args)
        return eval_dataset

    try:
        DatasetCatalog.register('instruction_dataset', load_instruction_dataset)
    except:
        print('dataset have been loaded')

    cont_id = eval_dataset.coco_id_to_cont_id if hasattr(eval_dataset,'coco_id_to_cont_id') else eval_dataset.common_id_to_cont_id
    class_name = eval_dataset.coco_class_name[:-1] if hasattr(eval_dataset,'coco_class_name') else eval_dataset.common_class_name
    ignore_label = 255 if not hasattr(eval_dataset,'ignore_label') else eval_dataset.ignore_label
    evaluator = my_SemSegEvaluator('instruction_dataset',
                                  output_dir=data_args.output_dir, dataset_id_to_cont_id=cont_id, class_name=class_name,ignore_label=ignore_label)
    evaluator.reset()


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compute_type=torch.float16
    model.to(dtype=torch.float16, device=device).eval()
    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            inputs = {k: v.to(device=device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            outputs = model.eval_seg(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                images=inputs['images'],
                seg_info=inputs['seg_info'],
                class_name_embedding_indices=inputs['class_name_embedding_indices'],
                class_name_ids=inputs['class_name_ids'],
                cls_indices=inputs['cls_indices'],
                labels=inputs['labels'],
                is_thing_list=eval_dataset.coco_is_thing if hasattr(eval_dataset,'coco_is_thing') else None
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if hasattr(eval_dataset,'common_class_name'):
                class_id_mapping = inputs['class_id_mapping'][0]
                sem_mask = torch.zeros(len(eval_dataset.common_class_name),outputs[0]['sem_seg'].shape[1],outputs[0]['sem_seg'].shape[2]).to(outputs[0]['sem_seg'].device)
                for i in range(outputs[0]['sem_seg'].shape[0]):
                    real_id = class_id_mapping[i]
                    sem_mask[real_id,:,:] = outputs[0]['sem_seg'][i,:,:]
                outputs = [{'sem_seg':sem_mask}]

            evaluator.process(inputs['seg_info'], outputs)

    results = evaluator.evaluate()
    if ov_task is not None:
        print(f'current ov_task is {ov_task}')
        print(results['sem_seg']['mIoU'])
    else:
        print(results)

    if results is None:
        results = {}
    return results





if __name__ == "__main__":
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    ov_task_list = data_args.ov_task_list
    if ov_task_list is None:
        evaluation(data_args)
    else:
        ov_task_list = ov_task_list.split('||')
        for ov_task in ov_task_list:
            evaluation(data_args,ov_task)
