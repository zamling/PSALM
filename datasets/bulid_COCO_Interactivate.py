import json
import os
from pycocotools.coco import COCO
from pycocotools.mask import encode, decode, frPyObjects
import numpy as np
from tqdm import tqdm
from skimage.measure import label, regionprops
from skimage.draw import line
from scipy.ndimage import gaussian_filter
import random


def calculate_iou(box1, box2):
    xA = max(box1[1], box2[1])
    yA = max(box1[0], box2[0])
    xB = min(box1[3], box2[3])
    yB = min(box1[2], box2[2])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

def generate_visual_prompt(mask):
    # point
    label_mask = label(mask)
    props_ = regionprops(label_mask)
    props = []
    for prop in props_:
        if prop.area > 5:
            props.append(prop)
    point_visual_prompt_mask = np.zeros_like(mask, dtype=np.uint8)

    for prop in props:
        # Randomly choose a point within the region
        min_row, min_col, max_row, max_col = prop.bbox
        centroid = prop.centroid
        for _ in range(1000):
            radius = min(prop.bbox[2] - prop.bbox[0], prop.bbox[3] - prop.bbox[1]) * 0.5
            angle = random.uniform(0, 2 * np.pi)
            offset = (random.uniform(0, radius) * np.cos(angle), random.uniform(0, radius) * np.sin(angle))
            point = (int(centroid[0] + offset[0]), int(centroid[1] + offset[1]))
            point = (np.clip(point[0], min_row, max_row - 1), np.clip(point[1], min_col, max_col - 1))
            if mask[point[0], point[1]] > 0:
                point_visual_prompt_mask[point[0], point[1]] = 1
                break

    # mask
    mask_visual_prompt_mask = gaussian_filter(mask.astype(float), sigma=2)
    mask_visual_prompt_mask = (mask_visual_prompt_mask > mask_visual_prompt_mask.mean()).astype(np.uint8)

    # box
    box_visual_prompt_mask = np.zeros_like(mask, dtype=np.uint8)

    for prop in props:
        min_row, min_col, max_row, max_col = prop.bbox
        scale_factor = random.uniform(0.9, 1.1)

        height = max_row - min_row
        width = max_col - min_col

        delta_height = height * (scale_factor - 1)
        delta_width = width * (scale_factor - 1)
        min_row = max(0, int(min_row - delta_height / 2))
        min_col = max(0, int(min_col - delta_width / 2))
        max_row = min(mask.shape[0], int(max_row + delta_height / 2))
        max_col = min(mask.shape[1], int(max_col + delta_width / 2))

        box_visual_prompt_mask[min_row:max_row, min_col:max_col] = 1

    # scribble
    scribble_visual_prompt_mask = np.zeros_like(mask, dtype=np.uint8)

    for prop in props:
        min_row, min_col, max_row, max_col = prop.bbox
        center_row, center_col = prop.centroid

        new_height = (max_row - min_row) * random.uniform(0.5, 1.2)
        new_width = (max_col - min_col) * random.uniform(0.5, 1.2)

        new_min_row = int(center_row - new_height / 2)
        new_min_col = int(center_col - new_width / 2)
        new_max_row = int(center_row + new_height / 2)
        new_max_col = int(center_col + new_width / 2)

        new_min_row, new_min_col = max(new_min_row, 0), max(new_min_col, 0)
        new_max_row, new_max_col = min(new_max_row, mask.shape[0]), min(new_max_col, mask.shape[1])

        new_box = (new_min_row, new_min_col, new_max_row, new_max_col)
        original_box = (min_row, min_col, max_row, max_col)
        flag = True

        for _ in range(1000):
            if calculate_iou(new_box, original_box) < 0.5:
                new_height = (max_row - min_row) * random.uniform(0.5, 1.2)
                new_width = (max_col - min_col) * random.uniform(0.5, 1.2)

                new_min_row = int(center_row - new_height / 2)
                new_min_col = int(center_col - new_width / 2)
                new_max_row = int(center_row + new_height / 2)
                new_max_col = int(center_col + new_width / 2)

                new_min_row, new_min_col = max(new_min_row, 0), max(new_min_col, 0)
                new_max_row, new_max_col = min(new_max_row, mask.shape[0]), min(new_max_col, mask.shape[1])

                new_box = (new_min_row, new_min_col, new_max_row, new_max_col)
            else:
                flag = False
                break
        if flag:
            continue

        corners = [(new_min_row, new_min_col), (new_min_row, new_max_col),
                   (new_max_row, new_min_col), (new_max_row, new_max_col)]

        start_point = random.choice(corners)
        corners.remove(start_point)
        if start_point in [(new_min_row, new_min_col), (new_max_row, new_max_col)]:
            end_point = (new_max_row if start_point[0] == new_min_row else new_min_row,
                         new_max_col if start_point[1] == new_min_col else new_min_col)
        else:
            end_point = (new_max_row if start_point[0] == new_min_row else new_min_row,
                         new_min_col if start_point[1] == new_max_col else new_max_col)

        rr, cc = line(start_point[0], start_point[1], end_point[0], end_point[1])
        rr = np.array(rr, dtype=np.float32)
        cc = np.array(cc, dtype=np.float32)

        amplitude = random.uniform(10, 20)
        frequency = random.uniform(0.2, 1)
        phase_shift = random.uniform(0, 2 * np.pi)
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * np.linspace(0, 1, len(rr)) + phase_shift)
        rr += sine_wave

        rr = np.clip(rr, 0, mask.shape[0] - 1).astype(np.int32)
        cc = np.clip(cc, 0, mask.shape[1] - 1).astype(np.int32)

        scribble_visual_prompt_mask[rr, cc] = 1

    return point_visual_prompt_mask, mask_visual_prompt_mask, box_visual_prompt_mask, scribble_visual_prompt_mask


if __name__ == '__main__':
    root_path = 'datasets/coco'
    splits = ['train', 'val']

    for split in splits:
        print(f'Processing {split}...')

        coco_path = os.path.join(root_path, f'annotation/instances_{split}2017.json')
        save_path = os.path.join(root_path, f'coco_interactive_{split}_psalm.json')
        coco = COCO(coco_path)

        coco_interactivate = []
        new_img_id = 0

        for img_id in tqdm(coco.imgs):
            img_info = coco.imgs[img_id]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            if len(anns) == 0:
                print('no annotation')
                continue
            for ann in anns:
                if isinstance(ann['segmentation'], list):
                    rle = frPyObjects(ann['segmentation'], img_info['height'], img_info['width'])
                    mask = coco.annToMask(ann)
                elif isinstance(ann['segmentation'], dict):
                    mask = coco.annToMask(ann)
                else:
                    raise ValueError("Unknown segmentation format")

                point_visual_prompt_mask, mask_visual_prompt_mask, box_visual_prompt_mask, scribble_visual_prompt_mask = generate_visual_prompt(
                    mask)
                point_rle = encode(np.asfortranarray(point_visual_prompt_mask))
                mask_rle = encode(np.asfortranarray(mask_visual_prompt_mask))
                box_rle = encode(np.asfortranarray(box_visual_prompt_mask))
                scribble_rle = encode(np.asfortranarray(scribble_visual_prompt_mask))
                ann['point_visual_prompt_mask'] = {
                    'counts': point_rle['counts'].decode('ascii'),
                    'size': point_rle['size']
                }
                ann['mask_visual_prompt_mask'] = {
                    'counts': mask_rle['counts'].decode('ascii'),
                    'size': mask_rle['size']
                }
                ann['box_visual_prompt_mask'] = {
                    'counts': box_rle['counts'].decode('ascii'),
                    'size': box_rle['size']
                }
                ann['scribble_visual_prompt_mask'] = {
                    'counts': scribble_rle['counts'].decode('ascii'),
                    'size': scribble_rle['size']
                }

            coco_interactivate.append({
                'image': img_info['file_name'],
                'image_info': img_info,
                'new_img_id': new_img_id,
                'anns': anns
            })
            new_img_id += 1

        with open(save_path, 'w') as f:
            json.dump(coco_interactivate, f, indent=2)

        print('dataset save in {}, max new_img_id: {}'.format(save_path, new_img_id))

