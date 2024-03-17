import json
from pycocotools.coco import COCO
from tqdm import tqdm
import concurrent.futures

if __name__ == '__main__':
    instrutions = {
        'Please segment all of objects in this image': [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    }
    coco_class_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        82, 84, 85, 86, 87, 88, 89, 90
    ]
    coco_id_to_cont_id = {coco_id: cont_id for cont_id, coco_id in enumerate(coco_class_ids)}

    splits = ['train','val']
    for split in splits:
        coco_path = 'datasets/coco/annotations/instances_{}2017.json'.format(split)
        print(coco_path)
        output_file = 'datasets/coco/instance_{}_psalm.json'.format(split)
        coco = COCO(coco_path)

        custom_dataset = []
        all_classes = set(class_name for classes in instrutions.values() for class_name in classes)

        new_img_id = 0

        for img_id in tqdm(coco.imgs):
            img_info = coco.imgs[img_id]

            class_to_anns = {}
            for class_name in all_classes:
                masks = []
                cat_id = coco.getCatIds(catNms=[class_name])
                ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_id)
                anns = coco.loadAnns(ann_ids)

                if anns:
                    class_to_anns[class_name] = anns

            for instruction, classes in instrutions.items():
                if any(class_name in class_to_anns for class_name in classes):
                    custom_dataset.append({
                        'image': img_info['file_name'],
                        'image_info': img_info,
                        'instruction': instruction,
                        'new_img_id': new_img_id,
                        'anns': [ann for class_name in classes if class_name in class_to_anns for ann in
                                 class_to_anns[class_name]],
                        'mask_classes': [class_name for class_name in classes if class_name in class_to_anns for mask in
                                         class_to_anns[class_name]],
                        'mask_classes_id': [coco.getCatIds(catNms=[class_name]) for class_name in classes if
                                            class_name in class_to_anns for mask in class_to_anns[class_name]]
                    })
                    new_img_id += 1
                else:
                    custom_dataset.append({
                        'image': img_info['file_name'],
                        'image_info': img_info,
                        'instruction': instruction,
                        'new_img_id': new_img_id,
                        'anns': [],
                        'mask_classes': [],
                        'mask_classes_id': []
                    })
                    new_img_id += 1

        with open(output_file, 'w') as f:
            json.dump(custom_dataset, f, indent=2)

        print('dataset save in {}, max new_img_id: {}'.format(output_file,new_img_id))
