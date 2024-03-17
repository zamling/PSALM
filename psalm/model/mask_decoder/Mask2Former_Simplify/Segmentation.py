from statistics import mode
from fvcore.common.config import CfgNode
import numpy as np
import os
import cv2
import glob
import tqdm
from PIL import Image
from PIL import ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from modeling.MaskFormerModel import MaskFormerModel
from utils.misc import load_parallal_model
from utils.misc import ADEVisualize

# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog

class Segmentation():
    def __init__(self, cfg, model=None):
        self.cfg = cfg
        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.size_divisibility = cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.device = torch.device("cuda", cfg.local_rank)        

        # data processing program
        self.padding_constant = 2**5 # resnet 总共下采样5次
        self.test_dir = cfg.TEST.TEST_DIR
        self.output_dir = cfg.TEST.SAVE_DIR
        self.imgMaxSize = cfg.INPUT.CROP.MAX_SIZE
        self.pixel_mean = np.array(cfg.DATASETS.PIXEL_MEAN)
        self.pixel_std = np.array(cfg.DATASETS.PIXEL_STD)
        self.visualize = ADEVisualize()        
        self.model = None

        pretrain_weights = cfg.MODEL.PRETRAINED_WEIGHTS
        if model is not None:
            self.model = model
        elif os.path.exists(pretrain_weights): 
            self.model = MaskFormerModel(cfg, is_init=False)
            self.load_model(pretrain_weights)
        else:
            print(f'please check weights file: {cfg.MODEL.PRETRAINED_WEIGHTS}')
        
    def load_model(self, pretrain_weights):
        state_dict = torch.load(pretrain_weights, map_location='cuda:0')

        ckpt_dict = state_dict['model']
        self.last_lr = state_dict['lr']
        self.start_epoch = state_dict['epoch']
        self.model = load_parallal_model(self.model, ckpt_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("loaded pretrain mode:{}".format(pretrain_weights))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.   
        img = (img - self.pixel_mean) / self.pixel_std
        img = img.transpose((2, 0, 1))
        return img

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def get_img_ratio(self, img_size, target_size):
        img_rate = np.max(img_size) / np.min(img_size)
        target_rate = np.max(target_size) / np.min(target_size)
        if img_rate > target_rate:
            # 按长边缩放
            ratio = max(target_size) / max(img_size)
        else:
            ratio = min(target_size) / min(img_size)
        return ratio

    def resize_padding(self, img, outsize, Interpolation=Image.BILINEAR):
        w, h = img.size
        target_w, target_h = outsize[0], outsize[1]
        ratio = self.get_img_ratio([w, h], outsize)
        ow, oh = round(w * ratio), round(h * ratio)
        img = img.resize((ow, oh), Interpolation)
        dh, dw = target_h - oh, target_w - ow
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)  # 左 顶 右 底 顺时针
        return img, [left, top, right, bottom]

    def get_img_ratio(self, img_size, target_size):
        img_rate = np.max(img_size) / np.min(img_size)
        target_rate = np.max(target_size) / np.min(target_size)
        if img_rate > target_rate:
            # 按长边缩放
            ratio = max(target_size) / max(img_size)
        else:
            ratio = min(target_size) / min(img_size)
        return ratio
    
    def image_preprocess(self, img):
        img_height, img_width = img.shape[0], img.shape[1]
        this_scale = self.get_img_ratio((img_width, img_height), self.imgMaxSize) # self.imgMaxSize / max(img_height, img_width)
        target_width = img_width * this_scale
        target_height = img_height * this_scale
        input_width = int(self.round2nearest_multiple(target_width, self.padding_constant))
        input_height = int(self.round2nearest_multiple(target_height, self.padding_constant))

        img, padding_info = self.resize_padding(Image.fromarray(img), (input_width, input_height))
        img = self.img_transform(img)

        transformer_info = {'padding_info': padding_info, 'scale': this_scale, 'input_size':(input_height, input_width)}
        input_tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
        return input_tensor, transformer_info

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[...,1:] 
        mask_pred = mask_pred.sigmoid()  
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)        
        return semseg.cpu().numpy()

    def postprocess(self, pred_mask, transformer_info, target_size):       
        oh, ow = pred_mask.shape[0], pred_mask.shape[1]
        padding_info = transformer_info['padding_info'] 
        
        left, top, right, bottom = padding_info[0], padding_info[1], padding_info[2], padding_info[3]
        mask = pred_mask[top: oh - bottom, left: ow - right]
        mask = cv2.resize(mask.astype(np.uint8), dsize=target_size, interpolation=cv2.INTER_NEAREST)
        return mask

    @torch.no_grad()
    def forward(self, img_list=None):        
        if img_list is None or len(img_list) == 0:
            img_list = glob.glob(self.test_dir + '/*.[jp][pn]g')
        mask_images = []
        for image_path in tqdm.tqdm(img_list):
            # img_name = os.path.basename(image_path)
            # seg_name = img_name.split('.')[0] + '_seg.png'
            # output_path = os.path.join(self.output_dir, seg_name)
            img = Image.open(image_path).convert('RGB')
            img_height, img_width = img.size[1], img.size[0]
            inpurt_tensor, transformer_info = self.image_preprocess(np.array(img))       

            outputs = self.model(inpurt_tensor)
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(inpurt_tensor.shape[-2], inpurt_tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks = self.semantic_inference(mask_cls_results, mask_pred_results)
            mask_img = np.argmax(pred_masks, axis=1)[0]
            mask_img = self.postprocess(mask_img, transformer_info, (img_width, img_height))
            mask_images.append(mask_img)
        return mask_images
                    

    def render_image(self, img, mask_img, output_path=None):
        self.visualize.show_result(img, mask_img, output_path)

        # ade20k_metadata = MetadataCatalog.get("ade20k_sem_seg_val")        
        # v = Visualizer(np.array(img), ade20k_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        # semantic_result = v.draw_sem_seg(mask_img).get_image()
        # if output_path is not None:
        #     cv2.imwrite(output_path, semantic_result)
        # else:
        #     cv2.imshow(semantic_result)
        #     cv2.waitKey(0)

