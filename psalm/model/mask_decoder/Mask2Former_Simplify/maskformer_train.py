#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   maskformer3D.py
@Time    :   2022/09/30 20:50:53
@Author  :   BQH 
@Version :   1.0
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   DeformTransAtten分割网络训练代码
'''

# here put the import lib

from statistics import mean
import torch
import numpy as np
import os
import time
import datetime
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch import distributed as dist
from torch.utils.data import DataLoader, SubsetRandomSampler
import sys
import math
import itertools
from PIL import Image
import wandb

from modeling.MaskFormerModel import MaskFormerModel
from utils.criterion import SetCriterion, Criterion
from utils.matcher import HungarianMatcher
from utils.summary import create_summary
from utils.solver import maybe_add_gradient_clipping
from utils.misc import load_parallal_model
from dataset.NuImages import NuImages
from Segmentation import Segmentation

class MaskFormer():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.size_divisibility = cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.device = torch.device("cuda", cfg.local_rank)
        self.is_training = cfg.MODEL.IS_TRAINING
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.last_lr = cfg.SOLVER.LR
        self.start_epoch = 0

        self.model = MaskFormerModel(cfg)
        if cfg.MODEL.PRETRAINED_WEIGHTS is not None and os.path.exists(cfg.MODEL.PRETRAINED_WEIGHTS):
            self.load_model(cfg.MODEL.PRETRAINED_WEIGHTS)
            print("loaded pretrain mode:{}".format(cfg.MODEL.PRETRAINED_WEIGHTS))

        self.model = self.model.to(self.device)
        if cfg.ngpus > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[cfg.local_rank], output_device=cfg.local_rank)             

        self._training_init(cfg)

        run_name = datetime.datetime.now().strftime("swin-%Y-%m-%d-%H-%M")
        self.run = wandb.init(
            project=cfg.project_name,
            name=run_name
        )
        wandb.watch(self.model)

    def build_optimizer(self):
        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim
            
        optimizer_type = self.cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                self.model.parameters(), self.last_lr, momentum=0.9, weight_decay=0.0001)
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                self.model.parameters(), self.last_lr)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

        if not self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(self.cfg, optimizer)

        return optimizer

    def load_model(self, pretrain_weights):
        state_dict = torch.load(pretrain_weights, map_location='cuda:0')
        print('loaded pretrained weights form %s !' % pretrain_weights)

        ckpt_dict = state_dict['model']
        self.last_lr = 6e-5 # state_dict['lr']
        self.start_epoch = 70 # state_dict['epoch']
        self.model = load_parallal_model(self.model, ckpt_dict)

    def _training_init(self, cfg):
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        boundary_weight = cfg.MODEL.MASK_FORMER.BOUNDARY_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]
        self.criterion = SetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            device=self.device
        )

        self.summary_writer = create_summary(0, log_dir=cfg.TRAIN.LOG_DIR)
        self.save_folder = cfg.TRAIN.CKPT_DIR
        self.optim = self.build_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='max', factor=0.9, patience=10)

    def reduce_mean(self, tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        return rt

    def train(self, train_sampler, data_loader, eval_loder, n_epochs):
        max_score = 0.88
        for epoch in range(self.start_epoch + 1, n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss = self.train_epoch(data_loader, epoch)
            evaluator_score = self.evaluate(eval_loder)
            evaluator_samples = self.evaluate_sample()
            self.scheduler.step(evaluator_score)
            # self.summary_writer.add_scalar('val_dice_score', evaluator_score, epoch)
            wandb.log({
                    "evaluator_score": evaluator_score,
                    "train_loss": train_loss,
                    "samples": [wandb.Image(sample) for sample in evaluator_samples],
                })
            if evaluator_score > max_score:
                max_score = evaluator_score
                ckpt_path = os.path.join(self.save_folder, 'mask2former_Epoch{0}_dice{1:.4f}.pth'.format(epoch, max_score))
                save_state = {'model': self.model.state_dict(),
                              'lr': self.optim.param_groups[0]['lr'],
                              'epoch': epoch}
                torch.save(save_state, ckpt_path)
                print('weights {0} saved success!'.format(ckpt_path))
        self.summary_writer.close()

    def train_epoch(self,data_loader, epoch):
        self.model.train()
        self.criterion.train()
        load_t0 = time.time()
        losses_list = []
        loss_ce_list = []
        loss_dice_list = []
        loss_mask_list = []
        
        for i, batch in enumerate(data_loader):                     
            inputs = batch['images'].to(device=self.device, non_blocking=True)
            targets = batch['masks']

            outputs = self.model(inputs)                
            losses = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
                        
            loss_ce = 0.0
            loss_dice = 0.0
            loss_mask = 0.0
            for k in list(losses.keys()):
                if k in weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                    if '_ce' in k:
                        loss_ce += losses[k]
                    elif '_dice' in k:
                        loss_dice += losses[k]
                    elif '_mask' in k:
                        loss_mask += losses[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            loss = loss_ce + loss_dice + loss_mask
            with torch.no_grad():
                losses_list.append(loss.item())
                loss_ce_list.append(loss_ce.item())
                loss_dice_list.append(loss_dice.item())
                loss_mask_list.append(loss_mask.item())

            self.model.zero_grad()
            self.criterion.zero_grad()
            loss.backward()
            loss = self.reduce_mean(loss, dist.get_world_size())
            self.optim.step()

            elapsed = int(time.time() - load_t0)
            eta = int(elapsed / (i + 1) * (len(data_loader) - (i + 1)))
            curent_lr = self.optim.param_groups[0]['lr']
            progress = f'\r[train] {i + 1}/{len(data_loader)} epoch:{epoch} {elapsed}(s) eta:{eta}(s) loss:{(np.mean(losses_list)):.6f} loss_ce:{(np.mean(loss_ce_list)):.6f} loss_dice:{(np.mean(loss_dice_list)):.6f} loss_mask:{(np.mean(loss_mask_list)):.6f}, lr:{curent_lr:.2e} '
            # progress = f'\r[train] {i + 1}/{len(data_loader)} epoch:{epoch} {elapsed}(s) eta:{eta}(s) loss:{(np.mean(losses_list)):.6f} loss_ce:{(np.mean(loss_ce_list)):.6f} loss_dice:{(np.mean(loss_dice_list)):.6f}, lr:{curent_lr:.2e}  '
            print(progress, end=' ')
            sys.stdout.flush()                
        
        self.summary_writer.add_scalar('loss', loss.item(), epoch)
        return loss.item()

    @torch.no_grad()                   
    def evaluate(self, eval_loder):
        self.model.eval()
        # self.criterion.eval()
        dice_score = []
        
        for batch in eval_loder:
            inpurt_tensor = batch['images'].to(device=self.device, non_blocking=True)
            gt_mask = batch['masks'][0]

            outputs = self.model(inpurt_tensor)
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            
            pred_masks = self.semantic_inference(mask_cls_results, mask_pred_results)                
            pred_mask = pred_masks[0]
            gt_binary_mask = self._get_binary_mask(gt_mask)
            dice = self._get_dice(pred_mask, gt_binary_mask.to(self.device))
            dice_score.append(dice.item())
        score = np.mean(dice_score)
        print('evaluate dice: {0}'.format(score))
        return score
    
    @torch.no_grad() 
    def evaluate_sample(self):        
        nuim = NuImages(dataroot=self.cfg.DATASETS.ROOT_DIR, version='v1.0-test') # v1.0-test or v1.0-mini
        sample_idx_list = np.random.choice(len(nuim.sample), 10, replace=False)
        seg_handler = Segmentation(self.cfg, self.model)
        input_imgs = []
        render_imgs = []
        for idx in sample_idx_list:
            sample = nuim.sample[idx]
            sd_token = sample['key_camera_token']
            sample_data = nuim.get('sample_data', sd_token)
            
            im_path = os.path.join(nuim.dataroot, sample_data['filename'])
            input_imgs.append(im_path)
        preds = seg_handler.forward(input_imgs)
        for i, img_path in enumerate(input_imgs):
            img = Image.open(img_path)
            render_img = nuim.render_predict(img, preds[i])
            render_imgs.append(render_img)
        return render_imgs

    def _get_dice(self, predict, target):    
        smooth = 1e-5    
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = predict.sum(-1) + target.sum(-1) 
        score = (2 * num + smooth).sum(-1) / (den + smooth).sum(-1)
        return score.mean()

    def _get_binary_mask(self, target):
        # 返回每类的binary mask
        y, x = target.size()
        target_onehot = torch.zeros(self.num_classes + 1, y, x)
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        return target_onehot[1:]

    def semantic_inference(self, mask_cls, mask_pred):       
        mask_cls = F.softmax(mask_cls, dim=-1)[...,1:]
        mask_pred = mask_pred.sigmoid()      
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)        
        return semseg

    # 实例分割待调试
    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result