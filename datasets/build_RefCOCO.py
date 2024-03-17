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
        sample_annotation = anno_id2info[sample['ann_id']]
        sample_image = img_id2info[sample['image_id']]
        outputs.append(
            {
                'image': sample_image['file_name'],
                'image_info': sample_image,
                'instruction': sample['sentences'],
                'anns': [sample_annotation],
                'new_img_id': new_img_id,
            }
        )
        new_img_id += 1

    with open(save_path, 'w') as f:
        json.dump(outputs, f)
    print(f'Saving at {save_path}. Total sample: {len(outputs)}.')

    

if __name__ == '__main__':
    # Change root path to your own directory
    root_path = 'datasets/refer_seg'
    datasets = ['refcoco', 'refcoco+, refcocog']
    splits = ['train', 'val', 'testA', 'testB']
    for dataset in datasets:
        if dataset == 'refcocog':
            splits = ['train', 'val', 'test']

        for split in splits:
            instance_path = os.path.join(root_path, f'{dataset}', 'instances.json')
            if dataset == 'refcocog':
                refs_name = 'refs(umd).p'
            else:
                refs_name = 'refs(unc).p'
            refs_path = os.path.join(root_path, f'{dataset}', refs_name)
            save_path = os.path.join(root_path, f'{dataset}', f'{split}_psalm.json')
            print(f'Processing {dataset}: {split}...')

            build_referring_dataset(instance_path, refs_path, split, save_path)

    print(f'Done')
