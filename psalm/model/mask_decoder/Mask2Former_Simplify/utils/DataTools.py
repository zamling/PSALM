#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DataTools.py
@Time    :   2022/09/30 07:47:36
@Author  :   zzubqh 
@Version :   1.0
@Contact :   baiqh@microport.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   数据集预处理类
'''

# here put the import lib
import os

def create_annotations():
    root_dir = r'/data/Dental'
    img_dir = os.path.join(root_dir, 'img')
    label_dir = os.path.join(root_dir, 'label')
    annotation = 'tooth_label.md'
    with open(annotation, 'w', encoding='utf-8') as wf:
        for img_file in os.listdir(img_dir):
            mask_file = os.path.join(label_dir, img_file.split('.')[0] + '_seg.nii.gz')
            if os.path.exists(mask_file):
                wf.write(f'{os.path.join(img_dir, img_file)},{mask_file}\r')

if __name__ == '__main__':
    create_annotations()
