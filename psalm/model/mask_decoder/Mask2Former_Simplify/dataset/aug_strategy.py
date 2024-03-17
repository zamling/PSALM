# !/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   CTPreprocess.py
@Time    :   2022/04/23 16:40:55
@Author  :   BQH 
@Version :   1.0
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   CT数据预处理
'''

# here put the import lib
import imgaug
import imgaug as ia
import numpy as np
import matplotlib.pyplot as plt
from imgaug import parameters as iap
import imgaug.augmenters as iaa
from matplotlib.patches import Rectangle
from imgaug.augmentables.bbs import BoundingBox
from imgaug.augmentables.bbs import BoundingBoxesOnImage

# 1随机参数
# 1.1正太分布 Normal 均值 or 方差 如果是StochasticParameter(sp)，每次调用func将对样本进行一次采样 2 采样 Choice([0.25, 0.5, 0.75], p=[0.25, 0.5, 0.25]) 以列表p概率对列表a中数值进行采样, 必须和a列表长度相同
# 在正态分布中，一个标准差范围所占比率为全部数值之68%，两个标准差之内的比率合起来为95%；三个标准差之内的比率合起来为99%。
sp_normal_rotate = iap.Normal(iap.Choice(a=[0, 90, 180, 270], p=[0.25, 0.25, 0.25, 0.25]), 22.5)  # 图像4个方向旋转增强,正态参数
sp_normal_translate = iap.Normal(loc=0, scale=0.3)
sp_normal_scale = iap.Normal(loc=1, scale=0.3)
sp_normal_hsv_v = iap.Normal(loc=0, scale=30)
sp_normal_shear = iap.Normal(loc=0, scale=4)
# 可视化
# iap.show_distributions_grid([sp_normal_scale])


# 2定义数据增强方式
# 使用iaa.Affine仿射变换 注意点1 cval` and `mode 处理 旋转移动等方式新产生的空白未定义区域 注意点2 涉及图像之间插值的使用order处理
# 2.1 仿射方式1 旋转 以图像中心为旋转轴，以度为单位的旋转,接受 数字 元组 列表 和 随机参数StochasticParameter对象，这里使用0 90 180 270 四峰值正太分布
meta_rotate = iaa.Affine(rotate=sp_normal_rotate)  # 旋转（影响热图）,作用到所有图像
# 2.2 平移
meta_translate = iaa.Affine(translate_percent={"x": sp_normal_translate, "y": sp_normal_translate})  # x和y不会相等
# 2.3 保持横纵向比例缩放
meta_scale = iaa.Affine(scale=sp_normal_scale)  # iaa.Scale不自动填充
# 2.4 镜像
meta_fliplr = iaa.Fliplr(0.5)
meta_flipud = iaa.Flipud(0.5)
# 2.5 图像模糊
meta_gblur = iaa.GaussianBlur(sigma=(0.0, 2.0))  # 0 is 无模糊 3 is 强模糊
meta_ablur = iaa.AverageBlur(k=((4, 10), (1, 3)))  # 模拟晃动模糊
meta_mblur = iaa.MedianBlur(k=(3, 11))  # 其他模糊
# 2.6 灰度化
meta_gray = iaa.Grayscale(alpha=(0.0, 1.0))
# 2.7 hsv调整
# meta_hsv_v = iaa.Sequential([
#     iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
#     iaa.WithChannels(channels=random.choice([0,1,2]), children=iaa.Add(sp_normal_hsv_v)),
#     iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
# ])
# WithChannels 中 children 必须是Augmenter增强器或者方法 不能是int或列表
meta_hsv = iaa.WithColorspace(
    to_colorspace="HSV",
    from_colorspace="RGB",
    children=iaa.OneOf([iaa.WithChannels(0, iaa.Multiply((0.6, 1.4))), iaa.WithChannels(1, iaa.Multiply((0.6, 1.4))),
                        iaa.WithChannels(2, iaa.Multiply((0.6, 1.4)))])
)
# 2.8 遮挡在transformations中实现
# iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
# iaa.Sharpen((0.0, 1.0)),       # 锐化图像,补偿图像的轮廓，增强图像的边缘及灰度跳变的部分，使图像变得清晰
# iaa.ElasticTransformation(alpha=50, sigma=5)  # 应用水效应（影响热图）
# 2.8 剪切变换
meta_shear = iaa.Affine(shear=sp_normal_shear)
# 2.9 分段扭曲
meta_pshear = iaa.PiecewiseAffine(scale=(0.01, 0.03))
# 2.10 4点透视变换
meta_ptransform = iaa.PerspectiveTransform(scale=(0.01, 0.20), keep_size=True)
# 2.11 对比度
meta_contrast_g = iaa.Sequential([iaa.GammaContrast((0.4, 1.6))])
meta_contrast_s = iaa.Sequential([iaa.SigmoidContrast(gain=(5, 20), cutoff=(0.25, 0.75))])

# 3数据增强序列管道
# 主要有两种 1.SomeOf应用数组增强序列中的0～n个, 2.Sometimes将数据增强序列应用于所有图片的%p,可以使用iaa.Sequential和SomeOf
# 拟采用Sometimes和SomeOf结合的方式，并随机顺序增强
# 3.1 作用到所有图像
pipe_sequential_rotate = iaa.Sequential(meta_rotate, random_order=True)
pipe_sequential_translate = iaa.Sequential(meta_translate, random_order=True)
pipe_sequential_scale = iaa.Sequential(meta_scale, random_order=True)
pipe_sequential_gray = iaa.Sequential(meta_gray, random_order=True)

# 3.2 随机使用0~n种增强方法
pipe_someof_flip = iaa.SomeOf((0, None), [meta_fliplr, meta_flipud], random_order=True)
pipe_someof_blur = iaa.SomeOf((0, 1), [meta_gblur, meta_ablur, meta_mblur], random_order=True)

# 3.3 增强p%的图像
pipe_sometimes_hsv = iaa.Sometimes(p=0.8, then_list=iaa.Sequential(meta_hsv, random_order=True))
pipe_sometimes_mpshear = iaa.Sometimes(p=0.5, then_list=iaa.OneOf([meta_shear, meta_pshear, meta_ptransform]))
# 3.5 其中一个
pipe_someone_contrast = iaa.OneOf([meta_contrast_g, meta_contrast_s])


def plt_box_on_img(ax, box, fill=False, alpha=0.5, color='r', linestyle='-'):
    rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=fill, alpha=alpha,
                     color=color, linestyle=linestyle)
    ax.add_patch(rect)


# 进行数据增强
def imgaug_mask(img, mask, seq):
    '''
    :param img: H W C int
    :param mask: H W int
    :param seq: aug_pipe
    :return:img H W C int mask H W int
    '''
    # 目前仅支持单个bbox
    # ia.seed(1)  使用此,同一种增强方式,所有的图片增强效果都相同,不建议使用
    # 创建与图像相关联的分割图的对象
    # 输入是H W or H W C,C=3通道或1通道都可以,结果是相同的
    # 建议使用int型数组,0代表背景,假设类不重叠,同时必须指定有多少类(包括背景),必须指定shape,shape是相应图片的形状.
    if isinstance(seq, imgaug.augmenters.meta.Augmenter):
        mask = ia.SegmentationMapsOnImage(mask, shape=img.shape)  # +1为背景
        seq_det = seq.to_deterministic()  # 将随机性增强转换为确定性增强,即使用相同的RandomState（即种子）开始每个批次.使用每个循环内图片增强方式不同,但是两个循环的图片增强的方式相同.建议使用,保持一致性,禁用将导致img和mask不对应
        img = seq_det.augment_image(img)  # 单张图片数据增强(H,W,C) ndarray or (H,W) ndarray
        mask = seq_det.augment_segmentation_maps([mask])[0]  # 多张图片mask数据增强，返回的是SegmentationMapsOnImage对象列表
        return img, mask.get_arr()
    else:
        return seq(np.array(img), np.array(mask, dtype=np.uint8))


def imgaug_boxes(img, boxes, seq):
    '''
    :param img: H W C int
    :param mask: H W int
    :param seq: aug_pipe
    :return:img H W C int mask H W int
    '''
    if isinstance(seq, imgaug.augmenters.meta.Augmenter):
        boxes = BoundingBoxesOnImage([BoundingBox(*i) for i in boxes], img.shape)
        seq_det = seq.to_deterministic()
        img = seq_det.augment_image(img)
        boxes = seq_det.augment_bounding_boxes([boxes, ])
        boxes = boxes[0].to_xyxy_array()
        h, w, c = img.shape
        return img, np.array([bbox_correct(i, w, h) for i in boxes])  # 仅支持图片上单个BBOX增强
    else:
        raise Exception("不支持的操作")


def imgaug_image(img, seq):
    '''
    :param img: H W C int
    :param mask: H W int
    :param seq: aug_pipe
    :return:img H W C int mask H W int
    '''
    if isinstance(seq, imgaug.augmenters.meta.Augmenter):
        seq_det = seq.to_deterministic()
        img = seq_det.augment_image(img)
        return img
    else:
        raise Exception("不支持的操作")


def bbox_correct(bbox, w, h):
    bbox = bbox.copy()
    bbox[0] = max(0, min(bbox[0], w))  # x0
    bbox[1] = max(0, min(bbox[1], h))  # y0
    bbox[2] = max(0, min(bbox[2], w))  # x1
    bbox[3] = max(0, min(bbox[3], h))  # y1
    return bbox


def mask_demo(image, mask, seq):
    for i in range(2):
        img, mask = imgaug_mask(image, mask, seq)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[1].imshow(mask)
        plt.show()


def boxes_demo(image, boxes, seq):
    for i in range(2):
        img_aug, boxes_aug = imgaug_boxes(image, boxes, seq)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        for i in boxes:
            plt_box_on_img(ax[0], i)
        ax[1].imshow(img_aug)
        for i in boxes_aug:
            plt_box_on_img(ax[1], i)
        plt.show()


if __name__ == '__main__':
    # Load an example image (uint8, 128x128x3). H W C int
    image = ia.quokka(size=(128, 128), extract="square")
    # load mask H W int
    segmap = np.zeros((128, 128), dtype=np.int8)
    segmap[28:71, 35:85] = 1
    boxes = [[32, 32, 64, 64], [64, 64, 100, 100]]
    # mask_demo(image, segmap, pipe_sequential_scale)
    for i in [iaa.Crop(px=(0, 10), keep_size=True)]:
        boxes_demo(image, boxes, i)