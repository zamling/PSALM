#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DatasetAnalyzer.py
@Time    :   2022/04/08 10:10:12
@Author  :   zzubqh 
@Version :   1.0
@Contact :   baiqh@microport.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib

import numpy as np
import os
import SimpleITK as sitk
from multiprocessing import Pool


class DatasetAnalyzer(object):
    """
    接收一个类似train.md的文件
    格式：**/ct_file.nii.gz, */seg_file.nii.gz
    """
    def __init__(self, annotation_file, num_processes=4):
        self.dataset = []
        self.num_processes = num_processes
        with open(annotation_file, 'r', encoding='utf-8') as rf:
            for line_item in rf:
                items = line_item.strip().split(',')
                self.dataset.append({'ct': items[0], 'mask': items[1]})

        print('total load {0} ct files'.format(len(self.dataset)))

    def _get_effective_data(self, dataset_item: dict):
        itk_img = sitk.ReadImage(dataset_item['ct'])
        itk_mask = sitk.ReadImage(dataset_item['mask'])

        img_np = sitk.GetArrayFromImage(itk_img)
        mask_np = sitk.GetArrayFromImage(itk_mask)

        mask_index = mask_np > 0
        effective_data = img_np[mask_index][::10]
        return list(effective_data)

    def compute_stats(self):
        if len(self.dataset) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        process_pool = Pool(self.num_processes)
        data_value = process_pool.map(self._get_effective_data, self.dataset)

        print('sub process end, get {0} case data'.format(len(data_value)))
        voxels = []
        for value in data_value:
            voxels += value

        median = np.median(voxels)
        mean = np.mean(voxels)
        sd = np.std(voxels)
        mn = np.min(voxels)
        mx = np.max(voxels)
        percentile_99_5 = np.percentile(voxels, 99.5)
        percentile_00_5 = np.percentile(voxels, 00.5)

        process_pool.close()
        process_pool.join()
        return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5


if __name__ == '__main__':
    import tqdm
    annotation = r'/home/code/Dental/Segmentation/dataset/tooth_label.md'
    analyzer = DatasetAnalyzer(annotation, num_processes=8)
    out_dir = r'/data/Dental/SegTrainingClipdata'
    # t = analyzer.compute_stats()
    # print(t)

    # new_annotation = r'/home/code/BoneSegLandmark/dataset/knee_clip_label_seg.md'
    # wf = open(new_annotation, 'w', encoding='utf-8')
    # with open(annotation, 'r', encoding='utf-8') as rf:
    #     for str_line in rf:
    #         items = str_line.strip().split(',')
    #         ct_name = os.path.basename(items[0])
    #         new_ct_path = os.path.join(out_dir, ct_name)
    #         label_file = items[1]
    #         wf.write('{0},{1}\r'.format(new_ct_path, label_file))
    # wf.close()

    # 根据CT值的范围重新生成新CT
    for item in tqdm.tqdm(analyzer.dataset):
        ct_file = item['ct']
        out_name = os.path.basename(ct_file)
        out_path = os.path.join(out_dir, out_name)
        itk_img = sitk.ReadImage(item['ct'])
        img_np = sitk.GetArrayFromImage(itk_img)
        data = np.clip(img_np, 181.0, 7578.0)
        clip_img = sitk.GetImageFromArray(data)
        clip_img.CopyInformation(itk_img)
        sitk.WriteImage(clip_img, out_path)

    

