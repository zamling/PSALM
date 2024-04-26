# Getting Started with PSALM

This document provides a brief introduction of the usage of PSALM.

For training and evaluation with PSALM, make sure you have prepared the corresponding dataset as [required](DATASET.md).

## Training in Command Line
PSALM conducts a two stage training strategy. 
- Download Siwn-B Mask2former from [here](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_54b88a.pkl)
- Download Phi-1.5 based on huggingface from [here](https://huggingface.co/susnato/phi-1_5_dev)

First stage training following [llava](https://github.com/haotian-liu/LLaVA/tree/v1.0.1?tab=readme-ov-file#pretrain-feature-alignment), the pretrained projector can be downloaded [here](https://huggingface.co/EnmingZhang/PSALM_stage1)

Second stage training:
- Run `bash scripts/train.sh` to train PSALM.
- Note: change model paths and dataset paths to the exact paths in `train.sh`


## Evaluation in Command Line
(Optional) Download our trained model from [here](../README.md#model-zoo)
### In-Domain Tasks
- **Panoptic COCO**
```
python psalm/eval/panoptic_segmentation.py --image_folder /path/to/coco/val2017/ --model_path /path/to/PSALM --json_path /path/to/coco
```
- **Instance COCO**
```
python psalm/eval/instance_segmentation.py --image_folder /path/to/coco/val2017/ --model_path /path/to/PSALM --json_path /path/to/coco/instance_val_psalm.json
```
- **RefCOCO**
```
python psalm/eval/referring_segmentation.py --image_folder /path/to/train2014 --json_path /path/to/refcoco/refcoco_val.json --model_path /path/to/PSALM
# also can eval refcoco+ and refcocog by replacing --json_path
```
- **Interactive COCO**
```
python psalm/eval/region_segmentation.py --image_folder /path/to/coco/val2017 --model_path /path/to/PSALM/ --json_path /path/to/coco_interactive_val_psalm.json --region_mask_type point_visual_prompt_mask
# also can eval box/scribble/mask by replacing --region_mask_type to box/scribble/mask_visual_prompt_mask
```
### Out-Domain Tasks
- **OV segmentation**
```
python psalm/eval/semantic_segmentation.py --model_path /path/to/PSALM --ov_task_list 'pc_20||ctx_459||ctx_59'
# you need to change the dataset paths in semantic_segmentation.py "OV_SEM_DICT"
```
- **gRefCOCO**
```
python psalm/eval/eval_grefcoco.py --model_path /path/to/PSALM --json_path /path/to/grefcoco/grefcoco_val.json --image_folder /path/to/train2014/
```
- **DAVIS**
```
python psalm/eval/eval_davis.py --image_folder /path/to/DAVIS --model_path /path/to/PSALM --json_path /path/to/DAVIS/2017/trainval_val_psalm.json
```
