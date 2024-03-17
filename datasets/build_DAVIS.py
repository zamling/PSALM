import json
import os
from PIL import Image
import numpy as np
from pycocotools.mask import encode, decode, frPyObjects
from tqdm import tqdm
import copy

if __name__ == '__main__':
    root_path = 'datasets/DAVIS'
    splits = ['trainval', 'test-dev']
    # we only do val evaluation
    annotation_path = os.path.join(root_path, f'2017/{splits[0]}/Annotations/480p')
    image_path = os.path.join(root_path, f'2017/{splits[0]}/JPEGImages/480p')

    set_path = os.path.join(root_path, f'2017/{splits[0]}/ImageSets/2017/val.txt')
    save_path = os.path.join(root_path, f'2017/{splits[0]}_val_psalm.json')

    val_set = []
    with open(set_path, 'r') as f:
        for line in f:
            val_set.append(line.strip())
    new_img_id = 0
    DAVIS_dataset = []
    for val_name in tqdm(val_set):
        vid_path = os.path.join(image_path, val_name)
        anno_path = os.path.join(annotation_path, val_name)

        first_frame_annotation_path = os.path.join(anno_path, sorted(os.listdir(anno_path))[0])
        first_frame_annotation_relpath = os.path.relpath(first_frame_annotation_path, root_path)

        first_frame_img_path = os.path.join(vid_path, sorted(os.listdir(vid_path))[0])
        first_frame_img_relpath = os.path.relpath(first_frame_img_path, root_path)

        first_frame_annotation_img = Image.open(first_frame_annotation_path)
        first_frame_annotation = np.array(first_frame_annotation_img)
        height, width = first_frame_annotation.shape
        unique_instances = np.unique(first_frame_annotation)
        unique_instances = unique_instances[unique_instances != 0]
        coco_format_annotations = []
        # for semi-supervised VOS, we use first frame's GT for input
        for instance_value in unique_instances:
            binary_mask = (first_frame_annotation == instance_value).astype(np.uint8)
            segmentation = encode(np.asfortranarray(binary_mask))
            segmentation = {
                'counts': segmentation['counts'].decode('ascii'),
                'size': segmentation['size'],
            }
            area = binary_mask.sum().astype(float)
            coco_format_annotations.append(
                {
                    'segmentation': segmentation,
                    'area': area,
                    'category_id': instance_value.astype(float),
                }
            )

        for filename, annfilename in zip(sorted(os.listdir(vid_path))[1:], sorted(os.listdir(anno_path))[1:]):
            sample_img_path = os.path.join(vid_path, filename)
            sample_img_relpath = os.path.relpath(sample_img_path, root_path)
            image_info = {
                'file_name': sample_img_relpath,
                'height': height,
                'width': width,
            }

            sample_annotation_path = os.path.join(anno_path, annfilename)
            sample_annotation = np.array(Image.open(sample_annotation_path))
            sample_unique_instances = np.unique(sample_annotation)
            sample_unique_instances = sample_unique_instances[sample_unique_instances != 0]
            anns = []
            for instance_value in sample_unique_instances:
                assert instance_value in unique_instances, 'Found new target not in the first frame'
                binary_mask = (sample_annotation == instance_value).astype(np.uint8)
                segmentation = encode(np.asfortranarray(binary_mask))
                segmentation = {
                    'counts': segmentation['counts'].decode('ascii'),
                    'size': segmentation['size'],
                }
                area = binary_mask.sum().astype(float)
                anns.append(
                    {
                        'segmentation': segmentation,
                        'area': area,
                        'category_id': instance_value.astype(float),
                    }
                )
            first_frame_anns = copy.deepcopy(coco_format_annotations)
            if len(anns) < len(first_frame_anns):
                first_frame_anns = [ann for ann in first_frame_anns if ann['category_id'] in sample_unique_instances]
            assert len(anns) == len(first_frame_anns)
            sample = {
                'image': sample_img_relpath,
                'image_info': image_info,
                'anns': anns,
                'first_frame_image': first_frame_img_relpath,
                'first_frame_anns': first_frame_anns,
                'new_img_id': new_img_id,
                'video_name': val_name,
            }
            DAVIS_dataset.append(sample)
            new_img_id += 1

    with open(save_path, 'w') as f:
        json.dump(DAVIS_dataset, f)
    print(f'Save at {save_path}. Total sample: {len(DAVIS_dataset)}')
