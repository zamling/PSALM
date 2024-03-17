# Update 2023.06.16
* "Modify the configs\Base-segmention.yaml file to add support for SwinTransformer
* Add configs\maskformer_nuimages.yaml to support the Nuimages dataset, and there is no need to install Nuimages Toolkit
* Add the wandb library to the training code, and you can see the training process on the Wandb website
<div align="center">
  <img src="https://github.com/zzubqh/Mask2Former-Simplify/raw/master/output/traing.png" width="100%" height="100%"/>
</div><br/>
# Description
* Remove the dependency on the detectron2 framework.
* Modify the data loading method to train by epoch instead of iteration, with a default of 300 epochs.
* Modify the data augmentation to use imgaug.
* Only the demo for semantic segmentation has been completed, and no instance segmentation code has been written.
* Only resnet50 has been trained as the backbone, and swin has not been debugged.

# Network architecture
backbone: resnet50\
Decoder: DefomTransformer + CrossAtten + SelfAtten

# Running Environment
* Inference Testing: Memory above 16GB; GPU VRAM above 4GB
* Network Training: Depending on the size of the image and the number of decoding layers, it is recommended to use 2 x 3090 GPUs with a batch size of 6 when the maximum image size is 512 and the number of decoding layers is 4.
# Usage
1. Install the packages listed in requirements.txt. Tested on Ubuntu 20.04 and should work on Windows as well.
2. Download the model [mask2former_resnet50](https://pan.baidu.com/s/16EsPxfn0L9ZoF-YtNY5KwA) with extraction code "usyz".
3. Copy the model to the "ckpt" folder in the project.
4. Copy the test images to the "test" folder or any other specified folder (if the user specifies a folder, configure the folder path in Base-segmention.yaml under TEST.TEST_DIR). The default results will be output to the "output" folder, but you can configure your own save directory using TEST.SAVE_DIR.
5. By default, the visualizer class from detectron2 is used for output display. If you do not want to install detectron2 or have problems after installation, you can also use the display method provided in the project, which differs from detectron2 in that it does not display the class names. To use the display method, modify Segmentation.py as follows: (1) comment out lines 17, 18, and 118 for importing and initializing detectron2; (2) uncomment line 144 for display; (3) comment out lines 145 to 147.(默认使用了detectron2中的visualizer类进行输出显示，如果不想安装detectron2或者安装后有问题，也可使用项目中的显示方式，与detectron2的区别在于没有对显示出类别名称，其余保持一致。对Segmentation.py进行如下修改：(1)注释掉17、18行和118行对detectron2包的引入和初始化；(2)放开第144行的显示调用；(3)注释掉145到147行即可)
```python
    mask_img = self.postprocess(mask_img, transformer_info, (img_width, img_height))
    self.visualize.show_result(img, mask_img, output_path)
    # v = Visualizer(np.array(img), ade20k_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    # semantic_result = v.draw_sem_seg(mask_img).get_image()
    # cv2.imwrite(output_path, semantic_result)     
```

<div align="center">
  <img src="https://github.com/zzubqh/Mask2Former-Simplify/raw/master/output/output.png" width="100%" height="100%"/>
</div><br/>

# Model Training
1. Prepare the dataset by downloading the ADEChallengeData2016 dataset and extracting it to a specified folder, such as: /home/xx/data
2. Configure the Base-segmention.yaml file by modifying DATASETS.ROOT_DIR to the folder where the dataset is located, such as: ROOT_DIR: '/home/dataset/'
3. Configure multi-scale training by modifying INPUT.CROP.SIZE and INPUT.CROP.MAX_SIZE in the Base-segmention.yaml file to the maximum image size during training. If hardware is limited, these values can be reduced.
4. Configure the number of transformer decoding layers by modifying MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS in the maskformer_ake150.yaml file. The default is 4, while the source code of the paper uses 6. This can be adjusted based on hardware conditions.
5. For multi-GPU training, specify the GPU number in main.py and run:
```
python -m torch.distributed.launch --nproc_per_node=2 main.py；
```
6. Single GPU training: Change the number in main.py to 0, and comment out line 213 in maskformer_train.py (used to average the loss across all GPUs). Single GPU training can be slow, it is recommended to use at least two GPUs.
# Preparing your own dataset
1. Store images and labels in two separate folders.
2. Write a *.odgt file, the format can be referred to as dataset/training.odgt.
3. Modify the DATASETS.TRAIN and DATASETS.VALID paths in the configuration file to the custom file paths, modify ROOT_DIR to the parent folder of the image and label files, and modify PIXEL_MEAN and PIXEL_STD to the mean and variance of the custom dataset.
4. Modify MODEL.SEM_SEG_HEAD.NUM_CLASSES in maskformer_ake150.yaml to the number of classes in the custom dataset, excluding the background class.

# Nuimage Dataset (Optional)
1. Download Dataset from [nuimages](https://www.nuscenes.org/nuimages)
2. The folder after decompression is shown as follows
```
|-- nuimages-v1.0-all-samples
    |-- samples
    |   |-- CAM_BACK
    |   |-- CAM_BACK_LEFT
    |   |-- CAM_BACK_RIGHT
    |   |-- CAM_FRONT
    |   |-- CAM_FRONT_LEFT
    |   `-- CAM_FRONT_RIGHT
    |-- v1.0-test
    |   |-- attribute.json
    |   |-- calibrated_sensor.json
    |   |-- category.json
    |   |-- ego_pose.json
    |   |-- log.json
    |   |-- object_ann.json
    |   |-- sample.json
    |   |-- sample_data.json
    |   |-- sensor.json
    |   `-- surface_ann.json
    |-- v1.0-train
    |   |-- attribute.json
    |   |-- calibrated_sensor.json
    |   |-- category.json
    |   |-- ego_pose.json
    |   |-- log.json
    |   |-- object_ann.json
    |   |-- sample.json
    |   |-- sample_data.json
    |   |-- sensor.json
    |   `-- surface_ann.json
    `-- v1.0-val
        |-- attribute.json
        |-- calibrated_sensor.json
        |-- category.json
        |-- ego_pose.json
        |-- log.json
        |-- object_ann.json
        |-- sample.json
        |-- sample_data.json
        |-- sensor.json
        `-- surface_ann.json
```
# Source Code Analysis
Please refer to the network structure and source code analysis：[Mask2Former源码解析](https://zhuanlan.zhihu.com/p/580645115)\
cite: https://github.com/facebookresearch/Mask2Former
