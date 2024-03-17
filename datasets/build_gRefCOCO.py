import json
import pickle
import os
from tqdm import tqdm

def build_referring_dataset(instance_path, refs_path, split, save_path):
    assert os.path.exists(instance_path), f'Path not found: {instance_path}'
    assert os.path.exists(refs_path), f'Path not found: {refs_path}'

    with open(instance_path) as f:
        instance = json.load(f)
    with open(refs_path, 'rb') as f:
        refs = pickle.load(f)

    images = instance['images']
    annotations = instance['annotations']

    img_id2info = {}
    for image in images:
        img_id2info[image['id']] = image
    anno_id2info = {}
    for annotation in annotations:
        anno_id2info[annotation['id']] = annotation

    outputs = []
    new_img_id = 0
    for sample in tqdm(refs):
        if sample['split'] != split:
            continue
        if -1 in sample['ann_id']:
            image = sample['file_name']
            image_id = sample['image_id']
            image_info = img_id2info[image_id]
            instruction = sample['sentences']
            ann_ids = sample['ann_id']
            anns = []
            result = {
                'image': image,
                'image_info': image_info,
                'instruction': instruction,
                'new_img_id': new_img_id,
                'anns': anns
            }
            outputs.append(result)
            new_img_id += 1
            continue
        image = sample['file_name']
        image_id = sample['image_id']
        image_info = img_id2info[image_id]
        instruction = sample['sentences']
        ann_ids = sample['ann_id']
        anns = [anno_id2info[id] for id in ann_ids]
        result = {
            'image': image,
            'image_info': image_info,
            'instruction': instruction,
            'new_img_id': new_img_id,
            'anns': anns
        }
        outputs.append(result)
        new_img_id += 1

    with open(save_path, 'w') as f:
        json.dump(outputs, f)
    print(f'Saving at {save_path}. Total sample: {len(outputs)}.')

if __name__ == '__main__':
    # Change root path to your own directory
    root_path = 'datasets/refer_seg'
    datasets = 'grefcoco'
    splits = ['train', 'val', 'testA', 'testB']

    for split in splits:
        instance_path = os.path.join(root_path, datasets, 'instances.json')
        refs_path = os.path.join(root_path, datasets, 'grefs(unc).json')
        save_path = os.path.join(root_path, datasets, f'{split}_psalm.json')
        print(f'Processing gRefCOCO: {split}...')

        build_referring_dataset(instance_path, refs_path, split, save_path)