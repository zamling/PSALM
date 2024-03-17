import json
import os
from pycocotools.coco import COCO
from tqdm import tqdm
import concurrent.futures
import re

if __name__ == '__main__':
    root_path = 'datasets/lvis'
    splits = ['train', 'val']
    for split in splits:
        print(f'Processing {split}.')
        annotation_path = os.path.join(root_path, f'lvis_v1_{split}.json')
        save_path = os.path.join(root_path, f'lvis_instance_psalm.json')

        with open(annotation_path) as f:
            data = json.load(f)
        categories = data['categories']
        categories_path = os.path.join(root_path, 'lvis_categories.json')
        with open(categories_path, 'w') as f:
            json.dump(categories, f, indent=2)

        annotation_map = {}
        for anno in data['annotations']:
            image_id = anno['image_id']
            if image_id not in annotation_map:
                annotation_map[image_id] = []
            annotation_map[image_id].append(anno)

        lvis_anno = []
        pattern = re.compile(r'.*/((?:train|val)\d+/\d+\.jpg)')
        new_img_id = 0
        for img in tqdm(data['images']):
            if img['id'] not in annotation_map:
                continue
            match = pattern.search(img['coco_url'])
            if match:
                image = match.group(1)
            else:
                image = img['coco_url']
            lvis_anno.append(
                {
                    'image': image,
                    'image_info': img,
                    'new_img_id': new_img_id,
                    'anns': annotation_map[img['id']]
                }
            )
            new_img_id += 1

        with open(save_path, 'w') as f:
            json.dump(lvis_anno, f, indent=2)
        print(f'saving at {save_path}.')

