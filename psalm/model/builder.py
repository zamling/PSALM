#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from psalm.model import *

from psalm.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from psalm.train.train_datasets import get_mask_config
from psalm.model.language_model.llava_phi import PSALM, PSALMForDAVISEval
def load_pretrained_model(model_path, model_base, model_name, model_args, mask_config='./psalm/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml', load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):

    kwargs = {"device_map": 'cpu'}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    print('loading segmentation model')
    model_map = {
        'psalm': PSALM,
        'psalm_video': PSALMForDAVISEval
    }
    model_map_name = model_args.model_map_name
    mask_cfg = get_mask_config(mask_config)
    mask_cfg.MODEL.MASK_FORMER.SEG_TASK = model_args.seg_task if hasattr(model_args, 'seg_task') else 'instance'

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    print(f'current model is {model_map_name}')
    model = model_map[model_map_name].from_pretrained(model_path, mask_decoder_cfg=mask_cfg, **kwargs)

    vision_tower = model.get_vision_tower()
    # if not vision_tower.is_loaded:
    #     vision_tower.load_model()
    # vision_tower.to(device=device, dtype=torch.float16)
    vision_tower.to(device=device)
    # if isinstance(vision_tower.image_processor,dict):
    #     image_processor = vision_tower.image_processor['instance']
    # else:
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
