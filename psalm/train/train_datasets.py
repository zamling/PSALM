import os
import random
import re
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import bisect
import torch
import numpy as np
import transformers

from psalm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_SEG_TOKEN, SEG_TOKEN_INDEX, DEFAULT_CLS_TOKEN, CLS_TOKEN_INDEX, DEFAULT_REGION_TOKEN, \
    REGION_TOKEN_INDEX, REFER_TOKEN_INDEX
from torch.utils.data import Dataset
from psalm.train.llava_trainer import LLaVATrainer

from psalm import conversation as conversation_lib
from psalm.model import *
from psalm.mm_utils import tokenizer_image_token

from PIL import Image

from psalm.mask_config.config import Config
from fvcore.common.config import CfgNode

from detectron2.structures import BoxMode
import warnings

warnings.filterwarnings('ignore')
local_rank = None

def get_mask_config(config='./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml'):
    cfg_coco = Config.fromfile(config)
    cfg_base = CfgNode.load_yaml_with_base(config, allow_unsafe=True)
    cfg_base.update(cfg_coco.__dict__.items())
    cfg = cfg_base
    cfg = Config(cfg)
    return cfg
class COCO_panoptic_dataset(Dataset):
    def __init__(self, json_path, tokenizer, data_args, is_train=True):
        super(COCO_panoptic_dataset).__init__()
        if is_train:
            self.panoptic_gt_path = os.path.join(json_path,'panoptic_train2017')
            self.panoptic_image_path = os.path.join(json_path,'train2017')
            self.panoptic_json_path = os.path.join(json_path,'annotations/panoptic_train2017.json')
            self.semantic_gt_path = os.path.join(json_path,'panoptic_semseg_train2017')
        else:
            self.panoptic_gt_path = os.path.join(json_path,'panoptic_val2017')
            self.panoptic_image_path = os.path.join(json_path,'val2017')
            self.panoptic_json_path = os.path.join(json_path,'annotations/panoptic_val2017.json')
            self.semantic_gt_path = os.path.join(json_path,'panoptic_semseg_val2017')

        with open(self.panoptic_json_path) as f:
            data = json.load(f)
        self.data = data['annotations']
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.mask_format = 'polygon'
        coco_class_ids = [cat['id'] for cat in data['categories']]
        coco_class_name = [cat['name'] for cat in data['categories']]
        coco_is_thing = [cat['isthing'] for cat in data['categories']]
        self.coco_id_to_cont_id = {coco_id: cont_id for cont_id, coco_id in enumerate(coco_class_ids)}
        self.coco_class_name = coco_class_name + ['background']
        self.coco_is_thing = coco_is_thing


    def __len__(self):
        return len(self.data)

    def preprocess_multimodal(self, sources):
        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                      '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')

                if DEFAULT_SEG_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_SEG_TOKEN, '').strip()
                    sentence['value'] = sentence['value'] + '\n' + DEFAULT_SEG_TOKEN
                    sentence['value'] = sentence['value']
        return sources

    def preprocess_llama2(self, sources, tokenizer):
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations

        input_ids = torch.stack(
            [self.tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        targets = input_ids.clone()

        assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

        # Mask targets
        sep = "[/INST] "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                round_len = len(self.tokenizer_special_tokens(rou, tokenizer))
                instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )

    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, cls_token_index=CLS_TOKEN_INDEX,
                                 region_token_index=REGION_TOKEN_INDEX, return_tensors=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<cls>': cls_token_index, '<region>':region_token_index}
        prompt_chunks = re.split('(<image>|<seg>|<cls>|<region>)', prompt)

        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            else:
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids

    def preprocess_class_name(self, CLS_token='[CAT]'):
        tokenized = [self.tokenizer.encode(class_name, add_special_tokens=False) for class_name in self.coco_class_name]
        tokenized_class_names = [tokens + [self.tokenizer.encode(CLS_token, add_special_tokens=False)[0]] for tokens in
                                 tokenized]
        class_name_id = [token for sublist in tokenized_class_names for token in sublist]
        class_name_id = torch.tensor(class_name_id)
        cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
        cls_indices = torch.tensor(cls_indices)

        return class_name_id, cls_indices

    def __getitem__(self, idx):
        data = self.data[idx]
        image_id = int(data["image_id"])
        image_file = os.path.join(self.panoptic_image_path, os.path.splitext(data["file_name"])[0] + ".jpg")

        data_dict = {}
        data_dict['file_name'] = image_file
        data_dict['image_id'] = image_id
        label_file = os.path.join(self.panoptic_gt_path, data["file_name"])
        sem_label_file = os.path.join(self.semantic_gt_path, data["file_name"])
        data_dict['pan_seg_file_name'] = label_file
        data_dict['sem_seg_file_name'] = sem_label_file
        segments_info = data["segments_info"]
        for seg in segments_info:
            seg['category_id'] = self.coco_id_to_cont_id[seg['category_id']]
        data_dict['segments_info'] = segments_info

        if isinstance(self.data_args.image_processor, dict):
            processor = self.data_args.image_processor['panoptic']
        else:
            processor = self.data_args.image_processor
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)
        instruction = 'Panoptic Segmentation: You need to segment all objects '
        prefix_inst = 'This is an image <image>, Please do Panoptic Segmentation.'

        num_class = len(self.coco_class_name)
        category = '<cls>, ' * (num_class-1) + '<cls>.'

        sources_value = f'\nThis is all the candidate categories: {category}\n'

        sources = [[{'from': 'human', 'value':  prefix_inst + sources_value},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]
        # sources = self.preprocess_multimodal(copy.deepcopy(sources))

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        class_name_ids, cls_indices = self.preprocess_class_name(CLS_token='[SEG]')
        class_name_embedding_indices = torch.zeros_like(input_ids)
        class_name_embedding_indices[input_ids == CLS_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]

        data_dict['class_name_ids'] = class_name_ids
        data_dict['cls_indices'] = cls_indices
        data_dict['class_name_embedding_indices'] = class_name_embedding_indices
        return data_dict

class COCO_interactive_dataset(COCO_panoptic_dataset):
    def __init__(self, json_path, tokenizer, data_args):
        if isinstance(json_path, list):
            data = []
            for path in json_path:
                with open(path) as f:
                    cur_data = json.load(f)
                data.extend(cur_data)
        else:
            with open(json_path) as f:
                data = json.load(f)
        self.data = data
        self.tokenizer = tokenizer
        self.data_args = data_args
        coco_class_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        coco_class_name = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.coco_id_to_cont_id = {coco_id: cont_id for cont_id, coco_id in enumerate(coco_class_ids)}
        self.coco_class_name = coco_class_name + ['background']
    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, cls_token_index=CLS_TOKEN_INDEX,
                                 region_token_index=REGION_TOKEN_INDEX, return_tensors=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<cls>': cls_token_index, '<region>':region_token_index}
        prompt_chunks = re.split('(<image>|<seg>|<cls>|<region>)', prompt)

        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            else:
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids

    def preprocess_class_name(self, CLS_token='[CAT]'):
        tokenized = [self.tokenizer.encode(class_name, add_special_tokens=False) for class_name in self.coco_class_name]
        tokenized_class_names = [tokens + [self.tokenizer.encode(CLS_token, add_special_tokens=False)[0]] for tokens in
                                 tokenized]
        # tokenized_class_names = [tokens for tokens in tokenized]
        class_name_id = [token for sublist in tokenized_class_names for token in sublist]
        class_name_id = torch.tensor(class_name_id)
        cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
        cls_indices = torch.tensor(cls_indices)

        return class_name_id, cls_indices
    def __getitem__(self, idx):
        data = self.data[idx]
        image_file = data['image']
        image_folder = self.data_args.image_folder


        data_dict = {}
        data_dict['file_name'] = os.path.join(image_folder, image_file)
        data_dict['height'] = data['image_info']['height']
        data_dict['width'] = data['image_info']['width']
        data_dict['image_id'] = data['new_img_id']
        data_dict['annotations'] = data['anns']
        for annotation in data_dict['annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            if annotation['category_id'] in self.coco_id_to_cont_id:
                annotation['category_id'] = self.coco_id_to_cont_id[annotation['category_id']]
            elif annotation['category_id'] in self.coco_id_to_cont_id.values():
                annotation['category_id'] = annotation['category_id']
            else:
                raise ValueError
            annotation['image_id'] = data['new_img_id']

        if isinstance(self.data_args.image_processor,dict):
            processor = self.data_args.image_processor['instance']
        else:
            processor = self.data_args.image_processor
        region_mask_type = getattr(self.data_args,'region_mask_type',None)
        if region_mask_type is not None:
            region_mask_type = region_mask_type.split('||')
        data_dict = processor.preprocess(data_dict,region_mask_type=region_mask_type)

        num_target = len(data_dict['instances'])
        prefix_inst = 'This is an image <image>, Please segment by given regions'
        regions_inst = ' <region>,' * (num_target - 1) + ' <region>.'
        sources_value = f'\nThis is all regions: {regions_inst}\n'

        sources = [
            [{'from': 'human', 'value': prefix_inst + sources_value},
             {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels
        data_dict['dataset_type'] = 'region_coco'

        return data_dict

class COCO_instance_dataset(COCO_interactive_dataset):
    def __init__(self, json_path, tokenizer, data_args):
        if isinstance(json_path, list):
            data = []
            for path in json_path:
                with open(path) as f:
                    cur_data = json.load(f)
                data.extend(cur_data)
        else:
            with open(json_path) as f:
                data = json.load(f)
        self.data = data
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.mask_format = 'polygon'
        coco_class_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90
        ]
        coco_class_name = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.coco_id_to_cont_id = {coco_id: cont_id for cont_id, coco_id in enumerate(coco_class_ids)}
        self.coco_class_name = coco_class_name + ['background']

    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, cls_token_index=CLS_TOKEN_INDEX,
                                 region_token_index=REGION_TOKEN_INDEX, return_tensors=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<cls>': cls_token_index, '<region>':region_token_index}
        prompt_chunks = re.split('(<image>|<seg>|<cls>|<region>)', prompt)

        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            else:
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids

    def preprocess_class_name(self, CLS_token='[CAT]'):
        tokenized = [self.tokenizer.encode(class_name, add_special_tokens=False) for class_name in self.coco_class_name]
        tokenized_class_names = [tokens + [self.tokenizer.encode(CLS_token, add_special_tokens=False)[0]] for tokens in
                                 tokenized]
        # tokenized_class_names = [tokens for tokens in tokenized]
        class_name_id = [token for sublist in tokenized_class_names for token in sublist]
        class_name_id = torch.tensor(class_name_id)
        cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
        cls_indices = torch.tensor(cls_indices)

        return class_name_id, cls_indices

    def __getitem__(self, idx):
        data = self.data[idx]
        image_file = data['image']
        image_folder = self.data_args.image_folder

        data_dict = {}
        data_dict['file_name'] = os.path.join(image_folder, image_file)
        data_dict['height'] = data['image_info']['height']
        data_dict['width'] = data['image_info']['width']
        data_dict['image_id'] = data['new_img_id']
        data_dict['annotations'] = data['anns']
        for annotation in data_dict['annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            if annotation['category_id'] in self.coco_id_to_cont_id:
                annotation['category_id'] = self.coco_id_to_cont_id[annotation['category_id']]
            elif annotation['category_id'] in self.coco_id_to_cont_id.values():
                annotation['category_id'] = annotation['category_id']
            else:
                raise ValueError
            annotation['image_id'] = data['new_img_id']

        if isinstance(self.data_args.image_processor, dict):
            processor = self.data_args.image_processor['instance']
        else:
            processor = self.data_args.image_processor
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)
        data_dict['annotations'] = data['anns']
        # instruction = data['instruction']
        instruction = 'Panoptic Segmentation: You need to segment all objects '
        prefix_inst = 'This is an image <image>, Please do Panoptic Segmentation.'

        num_class = len(self.coco_class_name)
        category = '<cls>, ' * (num_class - 1) + '<cls>.'

        sources_value = f'\nThis is all the candidate categories: {category}\n'

        sources = [[{'from': 'human', 'value':  prefix_inst + sources_value},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]

        # sources = self.preprocess_multimodal(copy.deepcopy(sources))

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        class_name_ids, cls_indices = self.preprocess_class_name(CLS_token='[SEG]')
        class_name_embedding_indices = torch.zeros_like(input_ids)
        class_name_embedding_indices[input_ids == CLS_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]

        data_dict['class_name_ids'] = class_name_ids
        data_dict['cls_indices'] = cls_indices
        data_dict['class_name_embedding_indices'] = class_name_embedding_indices
        return data_dict



class COCO_panoptic_dataset_random(COCO_panoptic_dataset):
    def preprocess_class_name(self, CLS_token='[CAT]'):
        random_idx = list(range(len(self.coco_class_name)))
        random.shuffle(random_idx)
        random_class_name = [self.coco_class_name[i] for i in random_idx]
        permute_idx = list(sorted(range(len(random_idx)), key=random_idx.__getitem__))
        tokenized = [self.tokenizer.encode(class_name, add_special_tokens=False) for class_name in random_class_name]
        tokenized_class_names = [tokens + [self.tokenizer.encode(CLS_token, add_special_tokens=False)[0]] for tokens in
                                 tokenized]
        class_name_id = [token for sublist in tokenized_class_names for token in sublist]
        class_name_id = torch.tensor(class_name_id)
        cls_indices = [idx for idx, sublist in enumerate(tokenized_class_names) for _ in sublist]
        cls_indices = torch.tensor(cls_indices)

        permute_idx = torch.tensor(permute_idx)


        return class_name_id, cls_indices, permute_idx

    def __getitem__(self, idx):
        data = self.data[idx]
        image_id = int(data["image_id"])
        image_file = os.path.join(self.panoptic_image_path, os.path.splitext(data["file_name"])[0] + ".jpg")

        data_dict = {}
        data_dict['file_name'] = image_file
        data_dict['image_id'] = image_id
        label_file = os.path.join(self.panoptic_gt_path, data["file_name"])
        sem_label_file = os.path.join(self.semantic_gt_path, data["file_name"])
        data_dict['pan_seg_file_name'] = label_file
        data_dict['sem_seg_file_name'] = sem_label_file
        segments_info = data["segments_info"]
        for seg in segments_info:
            if seg['category_id'] in self.coco_id_to_cont_id:
                seg['category_id'] = self.coco_id_to_cont_id[seg['category_id']]
            elif seg['category_id'] in self.coco_id_to_cont_id.values():
                seg['category_id'] = seg['category_id']
            else:
                raise ValueError
        data_dict['segments_info'] = segments_info



        processor = self.data_args.image_processor['panoptic']
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)
        # instruction = data['instruction']
        instruction = 'Panoptic Segmentation: You need to segment all objects '

        num_class = len(self.coco_class_name)
        category = '<cls>, ' * (num_class-1) + '<cls>.'

        sources_value = f'This is all the candidate categories: {category}\n<image>\n'

        sources = [[{'from': 'human', 'value': sources_value + instruction},
                    {'from': 'gpt', 'value': '\n[SEG]<seg>'}]]
        # sources = self.preprocess_multimodal(copy.deepcopy(sources))

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        class_name_ids, cls_indices, random_idx = self.preprocess_class_name()
        data_dict['random_idx'] = random_idx
        class_name_embedding_indices = torch.zeros_like(input_ids)
        class_name_embedding_indices[input_ids == CLS_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'panoptic_coco'

        data_dict['class_name_ids'] = class_name_ids
        data_dict['cls_indices'] = cls_indices
        data_dict['class_name_embedding_indices'] = class_name_embedding_indices
        return data_dict

class COCO_semantic_dataset(COCO_panoptic_dataset):
    def __getitem__(self, idx):
        data = self.data[idx]
        image_id = int(data["image_id"])
        image_file = os.path.join(self.panoptic_image_path, os.path.splitext(data["file_name"])[0] + ".jpg")

        data_dict = {}
        data_dict['file_name'] = image_file
        data_dict['image_id'] = image_id
        label_file = os.path.join(self.panoptic_gt_path, data["file_name"])
        sem_label_file = os.path.join(self.semantic_gt_path, data["file_name"])
        data_dict['pan_seg_file_name'] = sem_label_file
        data_dict['sem_seg_file_name'] = sem_label_file
        segments_info = data["segments_info"]
        for seg in segments_info:
            seg['category_id'] = self.coco_id_to_cont_id[seg['category_id']]
        data_dict['segments_info'] = segments_info

        if isinstance(self.data_args.image_processor, dict):
            processor = self.data_args.image_processor['panoptic']
        else:
            processor = self.data_args.image_processor
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)
        # instruction = data['instruction']
        instruction = 'Panoptic Segmentation: You need to segment all objects '
        prefix_inst = 'This is an image <image>, Please do Semantic Segmentation.'

        num_class = len(self.coco_class_name)
        category = '<cls>, ' * (num_class-1) + '<cls>.'

        sources_value = f'\nThis is all the candidate categories: {category}\n'

        sources = [[{'from': 'human', 'value':  prefix_inst + sources_value},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]
        # sources = self.preprocess_multimodal(copy.deepcopy(sources))

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        class_name_ids, cls_indices = self.preprocess_class_name(CLS_token='[SEG]')
        class_name_embedding_indices = torch.zeros_like(input_ids)
        class_name_embedding_indices[input_ids == CLS_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]

        data_dict['class_name_ids'] = class_name_ids
        data_dict['cls_indices'] = cls_indices
        data_dict['class_name_embedding_indices'] = class_name_embedding_indices
        return data_dict

class RefCOCO_dataset(COCO_instance_dataset):

    def preprocess_referring_instruction(self,instruction, REFER_token='[SEG]'):
        tokenized = self.tokenizer.encode(instruction, add_special_tokens=False)
        tokenized = tokenized + [self.tokenizer.encode(REFER_token, add_special_tokens=False)[0]]

        token_refer_id = torch.tensor(tokenized)

        return token_refer_id
    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, cls_token_index=CLS_TOKEN_INDEX,
                                 region_token_index=REGION_TOKEN_INDEX,refer_token_index=REFER_TOKEN_INDEX, return_tensors=None):
        input_ids = []
        special_token_map = {'<image>': image_token_index, '<seg>': seg_token_index, '<cls>': cls_token_index, '<region>':region_token_index, '<refer>':refer_token_index}
        prompt_chunks = re.split('(<image>|<seg>|<cls>|<region>|<refer>)', prompt)

        for chunk in prompt_chunks:
            if chunk in special_token_map:
                input_ids.append(special_token_map[chunk])
            else:
                input_ids.extend(tokenizer.encode(chunk, add_special_tokens=False))
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        else:
            return input_ids
    def __getitem__(self, idx):
        data = self.data[idx]
        image_file = data['image_info']['file_name']
        image_folder = self.data_args.refcoco_image_folder

        data_dict = {}
        data_dict['file_name'] = os.path.join(image_folder, image_file)
        data_dict['height'] = data['image_info']['height']
        data_dict['width'] = data['image_info']['width']
        data_dict['image_id'] = data['new_img_id']
        data_dict['annotations'] = data['anns']
        for annotation in data_dict['annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            # annotation['category_id'] = self.coco_id_to_cont_id[annotation['category_id']]
            if annotation['category_id'] in self.coco_id_to_cont_id:
                annotation['category_id'] = self.coco_id_to_cont_id[annotation['category_id']]
            elif annotation['category_id'] in self.coco_id_to_cont_id.values():
                annotation['category_id'] = annotation['category_id']
            else:
                raise ValueError
            annotation['image_id'] = data['new_img_id']

        if isinstance(self.data_args.image_processor,dict):
            processor = self.data_args.image_processor['instance']
        else:
            processor = self.data_args.image_processor
        data_dict = processor.preprocess(data_dict, mask_format=self.mask_format)
        # instruction = data['instruction']
        sentences = data['instruction']
        # prefix_inst = 'Referring Segmentation according to the following instruction:'
        prefix_inst = 'This is an image <image>, Please doing Referring Segmentation according to the following instruction:'
        instruction = ''
        for sent in sentences:
            instruction += ' {}.'.format(sent['sent'])
        sources = [[{'from': 'human', 'value': prefix_inst + '\n<refer>'},
                    {'from': 'gpt', 'value': '\nSure, the segmentation result is <seg>'}]]

        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]
        labels = text_dict['labels'][0]

        token_refer_id = self.preprocess_referring_instruction(instruction)
        refer_embedding_indices = torch.zeros_like(input_ids)
        refer_embedding_indices[input_ids == REFER_TOKEN_INDEX] = 1

        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'referring_coco'

        data_dict['token_refer_id'] = token_refer_id
        data_dict['refer_embedding_indices'] = refer_embedding_indices
        return data_dict

def preprocess_multimodal(
        sources,
        data_args
):
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                  '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

class UnifyDatasetSingleDatasetForBatch(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r


    def __init__(self,datasets,dataset_ratio,bs,fix_dataset_len=0):
        super(UnifyDatasetSingleDatasetForBatch, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.fix_dataset_len = fix_dataset_len

        self.cnt = 0
        self.bs = bs

        self.datasets = list(datasets)
        self.datasets_index_list = list(range(len(datasets)))
        self.dataset_ratio = dataset_ratio
        self.cur_dataset_index=0
        self.dataset_length = [len(data) for data in self.datasets]
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.coco_id_to_cont_id = {}
        self.coco_class_name = {}
        for _dataset in self.datasets:
            dataset_coco_id_to_cont_id = _dataset.coco_id_to_cont_id if hasattr(_dataset,'coco_id_to_cont_id') else []
            if len(dataset_coco_id_to_cont_id) > len(self.coco_id_to_cont_id):
                self.coco_id_to_cont_id = dataset_coco_id_to_cont_id
        for _dataset in self.datasets:
            _dataset.coco_id_to_cont_id = self.coco_id_to_cont_id
        for _dataset in self.datasets:
            dataset_coco_class_name = _dataset.coco_class_name if hasattr(_dataset,'coco_class_name') else []
            if len(dataset_coco_class_name) > len(self.coco_class_name):
                self.coco_class_name = dataset_coco_class_name
        for _dataset in self.datasets:
            _dataset.coco_class_name = self.coco_class_name
        # self.coco_id_to_cont_id = max([_dataset.coco_id_to_cont_id for _dataset in self.datasets])
        # for _dataset in self.datasets:
        #     _dataset.max_len = self.max_len
    def update_dataset_index(self):
        tempt = self.cur_dataset_index
        tempt += 1
        tempt = tempt % len(self.datasets)
        self.cur_dataset_index = tempt

    def __len__(self):
        if self.fix_dataset_len == 0:
            return self.cumulative_sizes[-1]
        else:
            return self.fix_dataset_len


    def __getitem__(self, idx):
        cur_dataset_len = self.dataset_length[self.cur_dataset_index]
        data_idx = idx % cur_dataset_len
        output_data = self.datasets[self.cur_dataset_index][data_idx]
        self.cnt += 1
        if self.cnt == self.bs:
            self.cnt = 0
            self.update_dataset_index()
        return output_data



class MM_Conv_Dataset(Dataset):
    def __init__(self, data_path,
                 tokenizer,
                 data_args):
        super(MM_Conv_Dataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict[:200000])
        # return len(self.list_data_dict)

    def preprocess_llama2(self, sources, tokenizer):
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations

        input_ids = torch.stack(
            [self.tokenizer_special_tokens(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        targets = input_ids.clone()

        assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

        # Mask targets
        sep = "[/INST] "
        idx = 0
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            if conv.version == 'phi':
                cur_len = 0
                target[:cur_len] = IGNORE_INDEX
                idx = 0
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    if idx > 0:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 2
                    else:
                        round_len = len(self.tokenizer_special_tokens(rou, tokenizer)) + 1
                    if idx > 0:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer))
                    else:
                        instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 1

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX
            else:
                cur_len = 1
                target[:cur_len] = IGNORE_INDEX
                for i, rou in enumerate(rounds):
                    if rou == "":
                        continue

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    round_len = len(self.tokenizer_special_tokens(rou, tokenizer))
                    instruction_len = len(self.tokenizer_special_tokens(parts[0], tokenizer)) - 2

                    target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                    cur_len += round_len
                    idx += 1
                target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids,
            labels=targets,
        )


    def tokenizer_special_tokens(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                 seg_token_index=SEG_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = []
        special_tokens = []
        image_splits = prompt.split('<image>')

        for i, chunk in enumerate(image_splits):
            if i != 0:
                special_tokens.append('<image>')
            seg_splits = chunk.split('<seg>')
            prompt_chunks.extend(seg_splits)
            special_tokens.extend(['<seg>'] * (len(seg_splits)-1))
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt_chunks]
        special_indexes = [image_token_index if token == '<image>' else seg_token_index for token in special_tokens]
        # easy one
        input_ids = []
        for i, chunk in enumerate(prompt_chunks):
            input_ids.extend(chunk)
            if i != len(prompt_chunks) -1:
                input_ids.extend([special_indexes[i]])
        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long).squeeze()
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        # if isinstance(i, int):
        #     sources = [sources]
        sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        data_dict = {}
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.mmconv_path
            if isinstance(self.data_args.image_processor, dict):
                processor = self.data_args.image_processor['instance']
            else:
                processor = self.data_args.image_processor
            if 'coco' in image_file:
                image_folder = self.data_args.image_folder
                image_file = os.path.basename(image_file)
                data_dict['file_name'] = os.path.join(image_folder, image_file)
            else:
                data_dict['file_name'] = os.path.join(image_folder, image_file)
            data_dict = processor.preprocess(data_dict)

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        data_dict['input_ids'] = text_dict['input_ids'][0]
        data_dict['labels'] = text_dict['labels'][0]
        data_dict['dataset_type'] = 'mm_conv'
        if 'image' not in data_dict:
            # image does not exist in the data, but the model is multimodal
            crop_size = 1024
            data_dict['image'] = torch.zeros(3, crop_size, crop_size)
        return data_dict
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

        return batch