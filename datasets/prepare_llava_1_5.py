import json
import cv2
import os
from tqdm import tqdm

image_root = '/path/to/llava/image/root'
coco_root = '/path/to/coco/train2017'
filter_list = []
with open('/path/to/llava/image/root/llava_v1_5_mix665k.json') as f:
    data = json.load(f)
for data_ in tqdm(data):
    if 'image' in data_:
        image_path = data_['image']
        if 'coco' in image_path:

            image_path = os.path.basename(image_path)
            file_name = os.path.join(coco_root, image_path)
        else:
            file_name = os.path.join(image_root, image_path)
        if os.path.exists(file_name):
            img = cv2.imread(file_name)
            if img is not None:
                filter_list.append(data_)
            else:
                print(f'cant open {file_name}')
        else:
            print(f'cant find {file_name}')
print(f'after filter, data length is {len(filter_list)}')
with open('/home/hk/yyma/data/mm_data_zem/LLaVA-Instruct-150K/llava_v1_5_mix665k_onlyMM_filtered.json','w') as f:
    json.dump(filter_list,f)
