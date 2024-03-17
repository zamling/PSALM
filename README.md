# PSALM: Pixelwise SegmentAtion with Large Multi-Modal Model
> #### Zheng Zhang\*, Yeyao Ma\*, Enming Zhang\*, Xiang Bai
>
> <sup>* Equal Contribution



### Features

* A powerful extension of the Large Multi-modal Model for generic (panoptic, instance, semantic) segmentation, referring segmentation and interactivate segmentation.
* Support joint training across multiple segmentation tasks and visual-language tasks.
* Demonstrates zero-shot capabilities on unseen task, such as open-vocabulary segmentation, generalizaed referring segmentation, and video object segmentation.

![teaser](images/teaser.png)

## Updates
- [x] Release evaluation code
- [x] Release training code
## Installation

See [Installation instructions](docs/INSTALL.md).

## Getting Started

See [Preparing Datasets for PSALM.](docs/DATASET.md)

See [Getting Started with PSALM.](docs/INSTALL.md)

## Model Zoo
- Download Mask2former [here](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_54b88a.pkl) firstly to get the pretrained image encoder.
- Download PSALM [here](https://huggingface.co/EnmingZhang/PSALM).
- Please change the `mm_vision_tower` to Mask2former checkpoint `*.pkl`.

## Citation
If you think this work is useful for your research, please use the following BibTeX entry.
```
{

}
```
## Acknowledgement
Thanks for awesome works: [Mask2former](https://github.com/facebookresearch/Mask2Former/tree/main), [Mask2former-Simplify](https://github.com/zzubqh/Mask2Former-Simplify)
and [LLaVA](https://github.com/haotian-liu/LLaVA). Code is based on these works.
