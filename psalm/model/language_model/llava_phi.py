from typing import List, Optional, Tuple, Union
from addict import Dict
from dataclasses import dataclass
import torch.nn.functional as F
import numpy as np
import pickle
import torch
import math
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from psalm.model.visual_prompt_module.context_cluster import region_pooling
from transformers import AutoConfig, AutoModelForCausalLM

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from psalm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, SEG_TOKEN_INDEX, CLS_TOKEN_INDEX, REGION_TOKEN_INDEX, REFER_TOKEN_INDEX
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import \
    MultiScaleMaskedTransformerDecoderForOPTPreTrain
from ..mask_decoder.Mask2Former_Simplify.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from ..multimodal_projector.builder import build_vision_projector
from ..multimodal_encoder.swin_trans import build_swin_b, build_swin_l


from ..datasets_mapper.coco_instance_mapper import COCOInstanceNewBaselineDatasetMapper
from ..datasets_mapper.coco_panoptic_mapper import COCOPanopticNewBaselineDatasetMapper
from ..datasets_mapper.coco_semantic_mapper import COCOSemanticNewBaselineDatasetMapper
from psalm.model.mask_decoder.mask_criterion.pretrain_criterion import PSALM_criterion, hungarian_matcher_PSALM
from transformers import PhiModel, PhiForCausalLM, PhiConfig
class LlavaConfig(PhiConfig):
    model_type = "llava_phi"

@dataclass
class CausalOutputWithMask(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_mask: Optional[torch.FloatTensor] = None
    loss_dice: Optional[torch.FloatTensor] = None
    loss_SEG_class: Optional[torch.FloatTensor] = None
    loss_class_name_class: Optional[torch.FloatTensor] = None
    loss_region_class: Optional[torch.FloatTensor] = None
    loss_llm: Optional[torch.FloatTensor] = None


class PSALMModel(LlavaMetaModel, PhiModel):
    config_class = LlavaConfig

    def __init__(self, config: PhiConfig, mask_decoder_cfg=None):
        super(PSALMModel, self).__init__(config)
        self.cfg = mask_decoder_cfg
        self.projector_outdim = config.hidden_size 
        if hasattr(config, "mm_vision_tower"):
            swin_type = getattr(config,'swin_type','base')
            if swin_type == 'base':
                self.vision_tower = build_swin_b(None)
            else:
                self.vision_tower = build_swin_l(None)
            self.mm_projector = build_vision_projector(config)
            self.vision_tower.image_processor = {}
            self.vision_tower.image_processor['panoptic'] = COCOPanopticNewBaselineDatasetMapper(self.cfg)
            self.vision_tower.image_processor['instance'] = COCOInstanceNewBaselineDatasetMapper(self.cfg)
            self.vision_tower.image_processor['semantic'] = COCOSemanticNewBaselineDatasetMapper(self.cfg)


    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower if hasattr(model_args, 'vision_tower') else model_args.mm_vision_tower
        with_norm = model_args.with_norm
        with_layernorm = model_args.with_layernorm
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter if hasattr(model_args,
                                                                                'pretrain_mm_mlp_adapter') else None
        projector_outdim = self.projector_outdim

        self.config.mm_vision_tower = vision_tower
        swin_type = getattr(model_args,'swin_type','base')
        self.config.swin_type = swin_type
        if swin_type == 'base':
            vision_tower = build_swin_b(vision_tower)
        else:
            print('current visual encoder is swin large')
            vision_tower = build_swin_l(vision_tower)
            self.config.mm_input_embeds = 1536

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower
        # if model_args.float32:
        #     print('convert sam parameters from fp16 to fp32')
        #     for name, module in self.vision_tower.named_modules():
        #         module = module.to(torch.float32)z

        self.config.use_mm_proj = True
        vision_tower.hidden_size = 256
        vision_tower.image_processor = {}
        vision_tower.image_processor['panoptic'] = COCOPanopticNewBaselineDatasetMapper(self.cfg)
        vision_tower.image_processor['instance'] = COCOInstanceNewBaselineDatasetMapper(self.cfg)
        vision_tower.image_processor['semantic'] = COCOSemanticNewBaselineDatasetMapper(self.cfg)
        # if model_args.seg_task == 'instance':
        #     vision_tower.image_processor = COCOInstanceNewBaselineDatasetMapper(self.cfg)
        # else:
        #     vision_tower.image_processor = COCOPanopticNewBaselineDatasetMapper(self.cfg)
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'conv')
        print(f'current mm_project_type is {self.config.mm_projector_type}, the output dim is {projector_outdim}')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.with_norm = with_norm
        self.config.with_layernorm = with_layernorm
        self.config.projector_outdim = projector_outdim

        if not hasattr(self, "mm_projector"):
            self.mm_projector = build_vision_projector(self.config)
        else:
            print('exist mm_projector, skip init')

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # import ipdb;ipdb.set_trace()
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)
            print('load mm_projector pth successfully')










class PSALM(PhiForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, mask_decoder_cfg=None, add_cross_attn=True, cross_attn_index=None):
        super(PSALM, self).__init__(config)

        self.model = PSALMModel(config, mask_decoder_cfg)
        self.init_config = config
        self.mask_decoder_cfg = mask_decoder_cfg
        self.cross_attn_index = cross_attn_index
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        is_train_mask_decode = getattr(config, 'mask_decode_train', False)
        self.is_train_mask_decode = is_train_mask_decode
        self.refer_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.class_name_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.region_sampler = region_pooling(num_sample_point=256)
        self.region_projector = nn.Linear(config.hidden_size, mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)

        if is_train_mask_decode:
            print('Mask Decoder has been trained, init directly')
            self.initial_mask_module()
        self.post_init()

    def initial_mask_module(self, pretrained_path=None, model_args=None):
        if not self.is_train_mask_decode:
            print('Initialize mask modules...')
            self.config.mask_decode_train = True
        self.seg_query = nn.Parameter(
            torch.zeros([self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES, self.config.hidden_size]))
        self.num_queries = self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.num_classes = self.mask_decoder_cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.test_topk_per_image = self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        input_shape = self.output_shape()
        self.pixel_decoder = self.pixel_decoder_init(cfg=self.mask_decoder_cfg, input_shape=input_shape)
        self.predictor = self.predictor_init(cfg=self.mask_decoder_cfg)

        self.seg_query_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
        self.SEG_token_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
        self.class_name_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)

        self.mask_decoder_training_init(self.mask_decoder_cfg)
        if pretrained_path is not None:
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            def change_w(weights, old_name, new_name):
                weights[new_name] = weights[old_name]
                weights.pop(old_name)

            if pretrained_path.endswith('.pkl'):
                with open(pretrained_path, 'rb') as f:
                    ckpt = pickle.load(f)
            else:
                ckpt = torch.load(pretrained_path)
            pixel_decoder_weights = get_w(ckpt['model'],'sem_seg_head.pixel_decoder')
            predictor_weights = get_w(ckpt['model'],'sem_seg_head.predictor')
            pixel_decoder_weights = {k: torch.tensor(v) for k, v in pixel_decoder_weights.items()}
            predictor_weights = {k: torch.tensor(v) for k, v in predictor_weights.items()}

            #deal some diff keys
            change_w(pixel_decoder_weights,'adapter_1.weight','adapter_1.0.weight')
            change_w(pixel_decoder_weights,'adapter_1.norm.weight','adapter_1.1.weight')
            change_w(pixel_decoder_weights,'adapter_1.norm.bias','adapter_1.1.bias')
            change_w(pixel_decoder_weights,'layer_1.weight','layer_1.0.weight')
            change_w(pixel_decoder_weights,'layer_1.norm.weight','layer_1.1.weight')
            change_w(pixel_decoder_weights,'layer_1.norm.bias','layer_1.1.bias')
            if 'static_query.weight' in predictor_weights:
                change_w(predictor_weights,'static_query.weight','query_feat.weight')
            if predictor_weights['query_embed.weight'].shape[0] == 200:
                predictor_weights['query_embed.weight'] = predictor_weights['query_embed.weight'][:100,:]
            diff_pixel_msg = self.pixel_decoder.load_state_dict(pixel_decoder_weights,strict=False)
            diff_predictor_msg = self.predictor.load_state_dict(predictor_weights,strict=False)
            print(diff_predictor_msg)
            print(diff_pixel_msg)


    def get_vision_tower_feature(self, images):
        features = self.get_model().get_vision_tower()(images)
        features_dict = {
            'res2': features[0],
            'res3': features[1],
            'res4': features[2],
            'res5': features[3],
        }
        return features_dict
    def mask_decoder_training_init(self, cfg):
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        # boundary_weight = cfg.MODEL.MASK_FORMER.BOUNDARY_WEIGHT

        matcher = hungarian_matcher_PSALM(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_SEG_class": class_weight, "loss_class_name_class": class_weight, "loss_mask": mask_weight,
                       "loss_dice": dice_weight, "loss_region_class": class_weight}
        self.weight_dict = weight_dict
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["SEG_labels", "class_name_labels", "masks", "region_labels"]
        self.criterion = PSALM_criterion(
            matcher=matcher,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            device=self.device
        )
        self.size_divisibility = 32
        if cfg.MODEL.MASK_FORMER.SEG_TASK == 'semantic':
            self.semantic_on = True
            self.instance_on = False
            self.panoptic_on = False
            self.referring_on = False
            self.region_on = False

        elif cfg.MODEL.MASK_FORMER.SEG_TASK == 'instance':
            self.semantic_on = False
            self.instance_on = True
            self.panoptic_on = False
            self.referring_on = False
            self.region_on = False
        elif cfg.MODEL.MASK_FORMER.SEG_TASK == 'panoptic':
            self.semantic_on = True
            self.instance_on = True
            self.panoptic_on = True
            self.referring_on = False
            self.region_on = False
        elif cfg.MODEL.MASK_FORMER.SEG_TASK == 'referring':
            self.semantic_on = False
            self.instance_on = False
            self.panoptic_on = False
            self.referring_on = True
            self.region_on = False
        elif cfg.MODEL.MASK_FORMER.SEG_TASK == 'region':
            self.semantic_on = False
            self.instance_on = False
            self.panoptic_on = False
            self.referring_on = False
            self.region_on = True
        else:
            raise NotImplementedError
        self.sem_seg_postprocess_before_inference = self.instance_on or self.panoptic_on or self.referring_on or self.region_on
    def get_region_embedding(self, hidden_states, region_embedding_masks):
        region_embedding_list = []
        for sample_hidden_satates, sample_region_embedding_masks in zip(hidden_states, region_embedding_masks):
            sample_region_embedding = sample_hidden_satates[sample_region_embedding_masks.bool()]
            region_embedding_list.append(sample_region_embedding)
        return region_embedding_list
    def SEG_instance_inference(self, SEG_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        scores = F.sigmoid(SEG_cls)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)

        mask_pred = mask_pred[topk_indices]

        result = Instances(image_size)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        return result
    def class_name_panoptic_inference(self, SEG_cls, class_name_cls, mask_pred):

        scores, labels = F.softmax(class_name_cls, dim=-1).max(-1)
        num_classes = class_name_cls.shape[-1] - 1
        mask_pred = mask_pred.sigmoid()

        object_mask_threshold = 0.8
        overlap_threshold = 0.8

        keep = labels.ne(num_classes) & (scores > object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = class_name_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = self.is_thing_list[pred_class]
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info
    def region_inference(self, region_cls, mask_pred):
        image_size = mask_pred.shape[-2:]

        scores = F.sigmoid(region_cls)


        result = Instances(image_size)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = (scores * mask_scores_per_image[None,...].repeat(scores.shape[0],1)).transpose(1,0)
        return result

    def class_name_semantic_inference(self, SEG_cls, class_name_cls, mask_pred):
        mask_cls = F.softmax(class_name_cls, dim=-1)[:, :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg
    def class_name_instance_inference(self, SEG_cls, class_name_cls, mask_pred):
        image_size = mask_pred.shape[-2:]

        cls_scores = F.softmax(class_name_cls, dim=-1)[:, :-1]
        scores = cls_scores

        num_classes = scores.shape[-1]

        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)

        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(5000, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        mask_pred = mask_pred[topk_indices]


        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = self.is_thing_list[lab]

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
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)  # [2,256,64,64]
        image_features = self.get_model().mm_projector(image_features[-1])
        return image_features

    def predictor_init(self, cfg):
        in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        nheads = cfg.MODEL.MASK_FORMER.NHEADS
        dim_feedforward = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        pre_norm = cfg.MODEL.MASK_FORMER.PRE_NORM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        enforce_input_project = False
        seg_norm = cfg.MODEL.MASK_FORMER.SEG_NORM
        seg_proj = cfg.MODEL.MASK_FORMER.SEG_PROJ
        seg_fuse_score = cfg.MODEL.MASK_FORMER.FUSE_SCORE
        seg_concat = False
        print(f'current seg concat mode: {seg_concat}, seg_norm: {seg_norm}, seg_proj: {seg_proj}, seg_fuse_score: {seg_fuse_score}')
        predictor = MultiScaleMaskedTransformerDecoderForOPTPreTrain(in_channels,
                                                                     hidden_dim,
                                                                     num_queries,
                                                                     nheads,
                                                                     dim_feedforward,
                                                                     dec_layers,
                                                                     pre_norm,
                                                                     mask_dim,
                                                                     enforce_input_project,
                                                                     seg_norm,
                                                                     seg_concat,
                                                                     seg_proj,
                                                                     seg_fuse_score)
        return predictor


    def get_model(self):
        return self.model
    def output_shape(self):
        out_features = self.mask_decoder_cfg.MODEL.SWIN.OUT_FEATURES
        out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        num_features = [int(self.mask_decoder_cfg.MODEL.SWIN.EMBED_DIM * 2 ** i) for i in
                        range(len(self.mask_decoder_cfg.MODEL.SWIN.DEPTHS))]
        out_feature_channels = {
            "res2": num_features[0],
            "res3": num_features[1],
            "res4": num_features[2],
            "res5": num_features[3],
        }
        backbone_feature_shape = dict()
        for name in out_features:
            backbone_feature_shape[name] = Dict(
                {'channel': out_feature_channels[name], 'stride': out_feature_strides[name]})
        return backbone_feature_shape

    def get_encoder_image(self, images):
        encode_image_features = self.get_model().get_vision_tower()(images)
        return encode_image_features

    def pixel_decoder_init(self, cfg, input_shape):
        common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        transformer_dropout = cfg.MODEL.MASK_FORMER.DROPOUT
        transformer_nheads = cfg.MODEL.MASK_FORMER.NHEADS
        transformer_dim_feedforward = 1024
        transformer_enc_layers = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        transformer_in_features = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES  # ["res3", "res4", "res5"]

        pixel_decoder = MSDeformAttnPixelDecoder(input_shape,
                                                 transformer_dropout,
                                                 transformer_nheads,
                                                 transformer_dim_feedforward,
                                                 transformer_enc_layers,
                                                 conv_dim,
                                                 mask_dim,
                                                 transformer_in_features,
                                                 common_stride)
        return pixel_decoder
    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def get_special_token(self, SEG, EOS):
        self.SEG_id = SEG
        self.EOS_id = EOS

    def get_class_name_embedding(self, hidden_states, cls_token_indices):
        class_name_embedding_list = []
        for current_hidden_state, current_token_indice in zip(hidden_states, cls_token_indices):
            class_id = torch.unique(current_token_indice)
            class_id = class_id[class_id != 0]
            current_class_name_embedding_list = []
            for id in class_id:
                current_class_mask = (current_token_indice == id)
                current_class_state = current_hidden_state[current_class_mask]
                current_class_name_embedding_list.append(current_class_state)
            current_pool_class_name_embedding = [self.class_name_pooling(class_name.transpose(-2, -1)).transpose(-2, -1)
                                                 for class_name in current_class_name_embedding_list]
            class_name_embedding_list.append(torch.cat(current_pool_class_name_embedding, dim=0))
        return torch.stack(class_name_embedding_list, dim=0)
    def embed_class_ids(self, class_name_ids, cls_indices):
        if class_name_ids is None:
            return None
        num_class = cls_indices.unique_consecutive()
        num_class = num_class[num_class >= 0]
        class_name_ids = [class_name_ids[cls_indices == idx] for idx in num_class]
        embedded_class_name = [self.get_model().embed_tokens(id) for id in class_name_ids]

        return embedded_class_name

    def embed_refer_ids(self, refer_ids):
        if refer_ids is None:
            return None
        embedded_refer = self.get_model().embed_tokens(refer_ids)
        return embedded_refer
    def concat_image_seg_cls_embeds(self, input_id, img_feature, label, seg_query, seg_query_mask, class_embed,
                                    class_name_embedding_indices,region_embedding_mask=None, region_feature_list=None, refer_embedding_indices=None,
                refer_embedding=None):
        image_token_indices = torch.where(input_id == IMAGE_TOKEN_INDEX)[0]
        seg_query_indices = torch.where(input_id == SEG_TOKEN_INDEX)[0]
        cls_token_indices = torch.where(input_id == CLS_TOKEN_INDEX)[0]
        region_token_indices = torch.where(input_id == REGION_TOKEN_INDEX)[0]
        assert len(image_token_indices) == 1, 'not supporting multi image index'
        assert len(seg_query_indices) == 1, 'not supporting multi seg index'
        if class_name_embedding_indices is not None:
            assert len(cls_token_indices) == len(class_embed), 'the number of <cls> tokens and class_embed needs to be same'
        if region_feature_list is not None:
            assert len(region_feature_list) == len(
                region_token_indices), 'the munber of <region> tokens and regions needs to be same'
        cur_new_input_embeds = []
        cur_new_seg_query_mask = []
        if label is not None:
            cur_new_label = []
            assert label.shape == input_id.shape
        else:
            cur_new_label = None
        cur_class_name_embedding_indices = [] if class_name_embedding_indices is not None else None
        cur_refer_embedding_indices = [] if refer_embedding_indices is not None else None

        if region_embedding_mask is not None:
            enable_region_mask = True
            cur_new_region_embedding_mask = []
        else:
            enable_region_mask = False
            cur_new_region_embedding_mask = None
        chunks = []
        current_chunk = []

        for id in input_id:
            if id >= 0:
                current_chunk.append(id.item())
            else:
                if current_chunk:
                    chunks.append(torch.tensor(current_chunk, device=input_id.device))
                    current_chunk = []
                chunks.append([id])
        if current_chunk:
            chunks.append(torch.tensor(current_chunk, device=input_id.device))

        cls_idx = 0
        region_idx = 0
        for chunk in chunks:
            chunk_len = len(chunk)
            if chunk_len == 1 and chunk[0] == IMAGE_TOKEN_INDEX:
                cur_new_input_embeds.append(img_feature)
                cur_new_seg_query_mask.append(torch.zeros(img_feature.shape[0]))
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(
                        torch.full((img_feature.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(
                        torch.full((img_feature.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(
                        torch.full((img_feature.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype)
                    )
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(torch.zeros(img_feature.shape[0]))
            elif chunk_len == 1 and chunk[0] == SEG_TOKEN_INDEX:
                cur_new_input_embeds.append(seg_query)
                cur_new_seg_query_mask.append(torch.ones(seg_query.shape[0]))
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(torch.full((seg_query.shape[0],), 0, device=label.device,
                                                                       dtype=label.dtype))
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(torch.full((seg_query.shape[0],), 0, device=label.device,
                                                                       dtype=label.dtype))
                if label is not None:
                    cur_new_label.append(
                        torch.full((seg_query.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype))
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(torch.zeros(seg_query.shape[0]))
            elif chunk_len == 1 and chunk[0] == CLS_TOKEN_INDEX:
                cls_embed = class_embed[cls_idx]
                if len(cls_embed.shape) == 1:
                    cls_embed = cls_embed.unsqueeze(0)
                cls_idx += 1
                cur_new_input_embeds.append(cls_embed)
                cur_new_seg_query_mask.append(torch.zeros(cls_embed.shape[0]))
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(torch.zeros(cls_embed.shape[0]))
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(
                        torch.full((cls_embed.shape[0],), cls_idx, device=input_id.device,
                                   dtype=input_id.dtype))
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(
                        torch.full((cls_embed.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(
                        torch.full((cls_embed.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype)
                    )
            elif chunk_len == 1 and chunk[0] == REGION_TOKEN_INDEX:
                region_feature = region_feature_list[region_idx]
                region_idx += 1
                cur_new_input_embeds.append(region_feature)
                cur_new_seg_query_mask.append(torch.zeros(region_feature.shape[0]))
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(
                        torch.full((region_feature.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(
                        torch.full((region_feature.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(
                        torch.full((region_feature.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype)
                    )
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(torch.ones(region_feature.shape[0]))
            elif chunk_len == 1 and chunk[0] == REFER_TOKEN_INDEX:
                refer_embed = refer_embedding
                if len(refer_embed.shape) == 1:
                    refer_embed = refer_embed.unsqueeze(0)
                cur_new_input_embeds.append(refer_embed)
                cur_new_seg_query_mask.append(torch.zeros(refer_embed.shape[0]))
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(torch.zeros(refer_embed.shape[0]))
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(
                        torch.full((refer_embed.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(
                        torch.full((refer_embed.shape[0],), 1, device=input_id.device,
                                   dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(
                        torch.full((refer_embed.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype)
                    )
            else:
                cur_new_input_embeds.append(self.get_model().embed_tokens(input_id[:chunk_len]))
                cur_new_seg_query_mask.append(seg_query_mask[:chunk_len])
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(class_name_embedding_indices[:chunk_len])
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(refer_embedding_indices[:chunk_len])
                if label is not None:
                    cur_new_label.append(label[:chunk_len])
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(region_embedding_mask[:chunk_len])

            input_id = input_id[chunk_len:]
            seg_query_mask = seg_query_mask[chunk_len:]
            if class_name_embedding_indices is not None:
                class_name_embedding_indices = class_name_embedding_indices[chunk_len:]
            if refer_embedding_indices is not None:
                refer_embedding_indices = refer_embedding_indices[chunk_len:]
            if label is not None:
                label = label[chunk_len:]
            if enable_region_mask:
                region_embedding_mask = region_embedding_mask[chunk_len:]

        cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
        if label is not None:
            cur_new_label = [x.to(device=self.device) for x in cur_new_label]
            cur_new_label = torch.cat(cur_new_label, dim=0)
        cur_new_seg_query_mask = [x.to(device=self.device) for x in cur_new_seg_query_mask]
        cur_new_seg_query_mask = torch.cat(cur_new_seg_query_mask, dim=0)
        if class_name_embedding_indices is not None:
            cur_class_name_embedding_indices = [x.to(device=self.device) for x in cur_class_name_embedding_indices]
            cur_class_name_embedding_indices = torch.cat(cur_class_name_embedding_indices, dim=0)
        if refer_embedding_indices is not None:
            cur_refer_embedding_indices = [x.to(device=self.device) for x in cur_refer_embedding_indices]
            cur_refer_embedding_indices = torch.cat(cur_refer_embedding_indices, dim=0)

        if enable_region_mask:
            cur_new_region_embedding_mask = [x.to(device=self.device) for x in cur_new_region_embedding_mask]
            cur_new_region_embedding_mask = torch.cat(cur_new_region_embedding_mask, dim=0)

        return cur_new_input_embeds, cur_new_label, cur_new_seg_query_mask, cur_class_name_embedding_indices, cur_new_region_embedding_mask, cur_refer_embedding_indices
    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, past_key_values, labels, images, class_name_embedding_indices=None,
            class_name_ids=None, cls_indices=None, instances=None, token_refer_id=None, refer_embedding_indices=None
    ):
        vision_tower = self.get_vision_tower()
        seg_query_mask = torch.zeros_like(input_ids)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, seg_query_mask

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        expanded_seg_query = self.seg_query.unsqueeze(0).expand(input_ids.shape[0], -1, -1)

        if (input_ids == REGION_TOKEN_INDEX).sum() != 0 and instances is not None:
            region_masks_list = [instance.region_masks.tensor for instance in instances]

            # [region_features_per_batch: [num_region, 1, dims]], len(region_features) = batch_size
            region_features = self.region_sampler(image_features, region_masks_list,
                                                  original_dtype=image_features.dtype,
                                                  return_dtype=image_features.dtype)
            region_embedding_masks = torch.zeros_like(input_ids)
        else:
            region_features = None
            region_embedding_masks = None
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_seg_query_masks = []
        new_class_name_embedding_indices = [] if class_name_embedding_indices is not None else None
        new_refer_embedding_indices = [] if refer_embedding_indices is not None else None
        new_region_embedding_masks = [] if region_features is not None else None
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_seg_query_mask = seg_query_mask[batch_idx]
            cur_seg_query = expanded_seg_query[batch_idx]
            cur_image_feature = image_features[batch_idx]
            cur_class_name_embedding_indices = class_name_embedding_indices[batch_idx] if class_name_embedding_indices is not None else None
            cur_refer_embedding_indices = refer_embedding_indices[batch_idx] if refer_embedding_indices is not None else None
            cur_region_feature_list = region_features[batch_idx] if region_features is not None else None
            cur_region_embedding_mask = region_embedding_masks[batch_idx] if region_features is not None else None
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # ensure gradients back propagation, not changing cur_input_embeds
                cur_input_embeds = cur_input_embeds + (
                        0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                new_seg_query_masks.append(cur_seg_query_mask)
                # cur_image_idx += 1
                continue

            if labels is not None:
                cur_label = labels[batch_idx]
            else:
                cur_label = None

            if class_name_ids is not None:
                cur_class_name_ids = class_name_ids[batch_idx]
                cur_cls_indices = cls_indices[batch_idx]
            else:
                cur_class_name_ids = None
                cur_cls_indices = None
            if token_refer_id is not None:
                cur_token_refer_id = token_refer_id[batch_idx]
            else:
                cur_token_refer_id = None


            cur_class_name_embedding = self.embed_class_ids(cur_class_name_ids, cur_cls_indices)
            cur_refer_embedding = self.embed_refer_ids(cur_token_refer_id)

            cur_input_embeds, cur_label, cur_seg_query_mask, cur_class_name_embedding_indices, cur_region_embedding_mask, cur_refer_embedding_indices = self.concat_image_seg_cls_embeds(
                input_id=cur_input_ids,
                img_feature=cur_image_feature,
                label=cur_label,
                seg_query=cur_seg_query,
                seg_query_mask=cur_seg_query_mask,
                class_embed=cur_class_name_embedding,
                class_name_embedding_indices=cur_class_name_embedding_indices,
                region_embedding_mask=cur_region_embedding_mask,
                region_feature_list=cur_region_feature_list,
                refer_embedding_indices=cur_refer_embedding_indices,
                refer_embedding=cur_refer_embedding
            )
            assert cur_input_embeds.shape[0] == cur_seg_query_mask.shape[0]

            new_input_embeds.append(cur_input_embeds)
            if labels is not None:
                new_labels.append(cur_label)
            new_seg_query_masks.append(cur_seg_query_mask)
            if class_name_embedding_indices is not None:
                new_class_name_embedding_indices.append(cur_class_name_embedding_indices)
            if refer_embedding_indices is not None:
                new_refer_embedding_indices.append(cur_refer_embedding_indices)
            if new_region_embedding_masks is not None:
                new_region_embedding_masks.append(cur_region_embedding_mask)
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)),
                                          dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            new_seg_query_masks_align = []
            for new_seg_query_mask in new_seg_query_masks:
                new_seg_query_mask = torch.cat(
                    (new_seg_query_mask, torch.zeros((max_len - new_seg_query_mask.shape[0]),dtype=new_seg_query_mask.dtype, device=new_seg_query_mask.device)),
                    dim=0)
                new_seg_query_masks_align.append(new_seg_query_mask)
            new_seg_query_masks = torch.stack(new_seg_query_masks_align, dim=0)

            new_class_name_embedding_indices_align = []

            if class_name_embedding_indices is not None:
                for new_class_name_embedding_indice in new_class_name_embedding_indices:
                    new_class_name_embedding_indice = torch.cat(
                        (new_class_name_embedding_indice,
                         torch.zeros((max_len - new_class_name_embedding_indice.shape[0]),dtype=new_class_name_embedding_indice.dtype, device=new_class_name_embedding_indice.device)),
                        dim=0)
                    new_class_name_embedding_indices_align.append(new_class_name_embedding_indice)
                new_class_name_embedding_indices = torch.stack(new_class_name_embedding_indices_align, dim=0)

            if refer_embedding_indices is not None:
                new_refer_embedding_indices_align = []
                for new_refer_embedding_indice in new_refer_embedding_indices:
                    new_refer_embedding_indice = torch.cat(
                        (new_refer_embedding_indice,
                         torch.zeros((max_len - new_refer_embedding_indice.shape[0]),dtype=new_refer_embedding_indice.dtype, device=new_refer_embedding_indice.device)),
                        dim=0)
                    new_refer_embedding_indices_align.append(new_refer_embedding_indice)
                new_refer_embedding_indices = torch.stack(new_refer_embedding_indices_align, dim=0)

            if new_region_embedding_masks is not None:
                new_region_embedding_masks_align = []
                for new_region_embedding_mask in new_region_embedding_masks:
                    new_region_embedding_mask = torch.cat(
                        (new_region_embedding_mask, torch.zeros((max_len - new_region_embedding_mask.shape[0]),dtype=new_region_embedding_mask.dtype, device=new_region_embedding_mask.device)),
                        dim=0)
                    new_region_embedding_masks_align.append(new_region_embedding_mask)
                new_region_embedding_masks = torch.stack(new_region_embedding_masks_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape

        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            new_seg_query_masks = torch.stack(new_seg_query_masks, dim=0)
            if class_name_embedding_indices is not None:
                new_class_name_embedding_indices = torch.stack(new_class_name_embedding_indices, dim=0)
            if refer_embedding_indices is not None:
                new_refer_embedding_indices = torch.stack(new_refer_embedding_indices, dim=0)

            if new_region_embedding_masks is not None:
                new_region_embedding_masks = torch.stack(new_region_embedding_masks, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, new_seg_query_masks, new_class_name_embedding_indices, new_region_embedding_masks, new_refer_embedding_indices
    def get_SEG_embedding(self,hidden_states, refer_embedding_indices):
        refer_embedding_list = []
        for current_hidden_state, current_token_indice in zip(hidden_states, refer_embedding_indices):
            current_refer_state = current_hidden_state[current_token_indice.bool()]
            current_pool_refer_state = self.refer_pooling(current_refer_state.transpose(-2, -1)).transpose(-2, -1)
            refer_embedding_list.append(current_pool_refer_state)
        return torch.stack(refer_embedding_list, dim=0)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            seg_info=None,
            class_name_ids=None,
            class_name_embedding_indices=None,
            cls_indices=None,
            random_idx=None,
            token_refer_id=None,
            refer_embedding_indices=None,
            dataset_type=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if dataset_type is not None:
            assert all(item == dataset_type[0] for item in dataset_type), f'this batch contain different dataset_type: {dataset_type}'
            batch_dataset_type = dataset_type[0]
            print(batch_dataset_type)
        else:
            batch_dataset_type = []
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids == SEG_TOKEN_INDEX).sum() != 0:
            if (input_ids == REGION_TOKEN_INDEX).sum() != 0:
                instances = [i['instances'] for i in seg_info]
            else:
                instances = None
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_query_mask, class_name_embedding_indices, region_embedding_masks, refer_embedding_indices = self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images, class_name_embedding_indices,
                class_name_ids, cls_indices, instances, token_refer_id, refer_embedding_indices)
        else:
            seg_query_mask = None
            class_name_embedding_indices = None
            region_embedding_masks = None
            SEG_token_indices = None
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.mm_conv_prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        if class_name_embedding_indices is not None:
            class_name_embedding = self.get_class_name_embedding(hidden_states, class_name_embedding_indices)
            class_name_embedding = self.class_name_projector(class_name_embedding)
        else:
            class_name_embedding = None

        if class_name_embedding is not None:
            class_name_embedding = torch.gather(class_name_embedding,dim=1,index=random_idx.unsqueeze(-1).repeat(1, 1, class_name_embedding.shape[-1]))

        if region_embedding_masks is not None:
            region_embedding_list = self.get_region_embedding(hidden_states, region_embedding_masks)
            region_embedding_list = [self.region_projector(region_embedding) for region_embedding in
                                     region_embedding_list]
        else:
            region_embedding_list = None
        if 'referring' in batch_dataset_type or 'region' in batch_dataset_type:
            class_name_embedding = None


        loss = None
        if labels is not None and seg_query_mask is None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            llm_loss = loss_fct(shift_logits, shift_labels)

        if seg_query_mask is not None:
            seg_query = self.get_seg_query(hidden_states, seg_query_mask)
            seg_query = self.seg_query_projector(seg_query)

            image_features = self.get_vision_tower_feature(images)
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
                image_features)

            if refer_embedding_indices is not None:
                SEG_embedding = self.get_SEG_embedding(hidden_states, refer_embedding_indices)
                SEG_embedding = self.SEG_token_projector(SEG_embedding)
            else:
                SEG_embedding = None
            if 'panoptic' in batch_dataset_type or 'region' in batch_dataset_type:
                SEG_embedding = None


            mask_outputs = self.predictor(multi_scale_features, mask_features, None, seg_query, SEG_embedding,
                                          class_name_embedding, region_embedding_list)
            if seg_info is not None:
                if "instances" in seg_info[0]:
                    gt_instances = [x["instances"].to(self.device) for x in seg_info]
                    # images = ImageList.from_tensors(images, self.size_divisibility)
                    targets = self.prepare_targets(gt_instances, images)
                else:
                    targets = None

                mask_losses = self.criterion(mask_outputs, targets)
                weight_dict = self.weight_dict

                loss_mask = 0.0
                loss_dice = 0.0
                loss_SEG_class = 0.0
                loss_class_name_class = 0.0
                loss_region_class = 0.0
                for k in list(mask_losses.keys()):
                    if k in weight_dict:
                        if mask_losses[k] is not None:
                            mask_losses[k] *= weight_dict[k]
                        if '_SEG' in k and mask_losses[k] is not None:
                            loss_SEG_class += mask_losses[k]
                        elif '_name' in k and mask_losses[k] is not None:
                            loss_class_name_class += mask_losses[k]
                        elif '_mask' in k:
                            loss_mask += mask_losses[k]
                        elif '_dice' in k:
                            loss_dice += mask_losses[k]
                        elif '_region' in k and mask_losses[k] is not None:
                            loss_region_class += mask_losses[k]
                    else:
                        mask_losses.pop(k)
                mask_loss = loss_mask + loss_dice + loss_SEG_class + loss_class_name_class + loss_region_class
                if isinstance(loss_class_name_class, float):
                    loss_class_name_class = torch.tensor(loss_class_name_class, device=mask_loss.device)
                if isinstance(loss_SEG_class, float):
                    loss_SEG_class = torch.tensor(loss_SEG_class, device=mask_loss.device)
                if isinstance(loss_region_class, float):
                    loss_region_class = torch.tensor(loss_region_class, device=mask_loss.device)
            llm = torch.tensor(0.0, device=mask_loss.device)
            if labels is not None:
                # loss = llm_loss + mask_loss
                loss = mask_loss

            return CausalOutputWithMask(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                loss_mask=loss_mask.detach(),
                loss_dice=loss_dice.detach(),
                loss_SEG_class=loss_SEG_class.detach(),
                loss_class_name_class=loss_class_name_class.detach(),
                loss_region_class=loss_region_class.detach(),
                loss_llm=llm.detach(),
            )

        if labels is not None and seg_query_mask is None:
            loss_mask = torch.tensor(0.0, device=llm_loss.device)
            loss_dice = torch.tensor(0.0, device=llm_loss.device)
            loss_SEG_class = torch.tensor(0.0, device=llm_loss.device)
            loss_class_name_class = torch.tensor(0.0, device=llm_loss.device)
            loss_region_class = torch.tensor(0.0, device=llm_loss.device)
            loss = llm_loss
        else:
            return CausalOutputWithMask(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        return CausalOutputWithMask(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_mask=loss_mask.detach(),
            loss_dice=loss_dice.detach(),
            loss_SEG_class=loss_SEG_class.detach(),
            loss_class_name_class=loss_class_name_class.detach(),
            loss_region_class=loss_region_class.detach(),
            loss_llm=llm_loss.detach(),
        )

    def mm_conv_prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # ensure gradients back propagation, not changing cur_input_embeds
                cur_input_embeds = cur_input_embeds + (0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            # concat text and image embedding. prepare labels, IGNORE_INDEX for image tokens
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # Align embedddings, labels, attn_mask from different sample into a batch
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def get_seg_query(self, hidden_states, seg_query_masks):
        seg_query_list = []
        for sample_hidden_state, sample_query_mask in zip(hidden_states, seg_query_masks):
            if torch.sum(sample_query_mask) == 0:
                continue

            unique_query_value = torch.unique(sample_query_mask)
            unique_query_value = unique_query_value[unique_query_value != 0]

            for value in unique_query_value:
                current_query_mask = (sample_query_mask == value)
                current_query = sample_hidden_state[current_query_mask]

                seg_query_list.append(current_query)

        seg_query = torch.stack(seg_query_list, dim=0)

        return seg_query
    def eval_seg(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            seg_info=None,
            class_name_ids=None,
            class_name_embedding_indices=None,
            cls_indices=None,
            token_refer_id=None,
            refer_embedding_indices=None,
            is_thing_list=None
    ):
        if self.panoptic_on:
            assert is_thing_list is not None, 'is_thing_list need to be given'
            self.is_thing_list = is_thing_list
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids == REGION_TOKEN_INDEX).sum() != 0:
            instances = [i['instances'] for i in seg_info]
        else:
            instances = None
        input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_query_mask, class_name_embedding_indices, region_embedding_masks, refer_embedding_indices = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images, class_name_embedding_indices,
            class_name_ids, cls_indices, instances, token_refer_id, refer_embedding_indices)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs.last_hidden_state
        seg_query = self.get_seg_query(hidden_states, seg_query_mask)
        seg_query = self.seg_query_projector(seg_query)

        image_features = self.get_vision_tower_feature(images)
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features)

        if refer_embedding_indices is not None:
            SEG_embedding = self.get_SEG_embedding(hidden_states, refer_embedding_indices)
            SEG_embedding = self.SEG_token_projector(SEG_embedding)
        else:
            SEG_embedding = None

        if class_name_embedding_indices is not None:
            class_name_embedding = self.get_class_name_embedding(hidden_states, class_name_embedding_indices)
            class_name_embedding = self.class_name_projector(class_name_embedding)
        else:
            class_name_embedding = None

        if region_embedding_masks is not None:
            region_embedding_list = self.get_region_embedding(hidden_states, region_embedding_masks)
            region_embedding_list = [self.region_projector(region_embedding) for region_embedding in
                                     region_embedding_list]
        else:
            region_embedding_list = None

        mask_outputs = self.predictor(multi_scale_features, mask_features, None, seg_query, SEG_embedding,
                                      class_name_embedding, region_embedding_list)

        SEG_cls_results = mask_outputs['pred_SEG_logits']
        class_name_cls_results = mask_outputs['pred_class_name_logits']
        mask_pred_results = mask_outputs["pred_masks"]
        region_cls_results = mask_outputs['pred_region_logits']
        images = [x for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        del mask_outputs
        processed_results = []
        if SEG_cls_results is None:
            SEG_cls_results = [None]
        if class_name_cls_results is None:
            class_name_cls_results = [None]
        for _seg_info, SEG_cls_result, class_name_cls_result, mask_pred_result, input_per_image, image_size in zip(
                seg_info, SEG_cls_results, class_name_cls_results, mask_pred_results, seg_info, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            padding_mask = input_per_image.get("padding_mask")
            non_padding_indices = np.where(~ np.array(padding_mask))
            min_y, max_y = np.min(non_padding_indices[0]), np.max(non_padding_indices[0])
            min_x, max_x = np.min(non_padding_indices[1]), np.max(non_padding_indices[1])
            original_height = max_y - min_y + 1
            original_width = max_x - min_x + 1
            processed_results.append({})
            # gt = _seg_info['instances'].gt_masks
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, [original_height, original_width], height, width
                )
                if SEG_cls_result is not None:
                    SEG_cls_result = SEG_cls_result.to(mask_pred_result)

            if self.semantic_on:
                semantic_r = retry_if_cuda_oom(self.class_name_semantic_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                if not self.sem_seg_postprocess_before_inference:
                    semantic_r = retry_if_cuda_oom(sem_seg_postprocess)(
                    semantic_r, [original_height, original_width], height, width
                )
                processed_results[-1]["sem_seg"] = semantic_r

            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.class_name_instance_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.class_name_panoptic_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                processed_results[-1]["panoptic_seg"] = panoptic_r
            if self.referring_on:
                instance_r = retry_if_cuda_oom(self.SEG_instance_inference)(SEG_cls_result.float(),
                                                                            mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
            if self.region_on:
                gt = _seg_info['instances'].gt_masks
                gt_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    gt, [original_height, original_width], height, width
                )
                region_cls_results = region_cls_results[0].to(mask_pred_result)
                instance_r = retry_if_cuda_oom(self.region_inference)(region_cls_results.float(),
                                                                            mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
                processed_results[-1]["gt"] = gt_result





            return processed_results




class PSALMForDAVISEval(PSALM):
    def eval_seg(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            seg_info=None,
            class_name_ids=None,
            class_name_embedding_indices=None,
            cls_indices=None,
            token_refer_id=None,
            refer_embedding_indices=None,
            is_thing_list=None,
            vp_images=None
    ):
        if self.panoptic_on:
            assert is_thing_list is not None, 'is_thing_list need to be given'
            self.is_thing_list = is_thing_list
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids == REGION_TOKEN_INDEX).sum() != 0:
            instances = [i['instances'] for i in seg_info]
        else:
            instances = None
        input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_query_mask, class_name_embedding_indices, region_embedding_masks, refer_embedding_indices = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images,vp_images, class_name_embedding_indices,
            class_name_ids, cls_indices, instances, token_refer_id, refer_embedding_indices)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs.last_hidden_state
        seg_query = self.get_seg_query(hidden_states, seg_query_mask)
        seg_query = self.seg_query_projector(seg_query)

        image_features = self.get_vision_tower_feature(images)
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features)

        if refer_embedding_indices is not None:
            SEG_embedding = self.get_SEG_embedding(hidden_states, refer_embedding_indices)
            SEG_embedding = self.SEG_token_projector(SEG_embedding)
        else:
            SEG_embedding = None

        if class_name_embedding_indices is not None:
            class_name_embedding = self.get_class_name_embedding(hidden_states, class_name_embedding_indices)
            class_name_embedding = self.class_name_projector(class_name_embedding)
        else:
            class_name_embedding = None

        if region_embedding_masks is not None:
            region_embedding_list = self.get_region_embedding(hidden_states, region_embedding_masks)
            region_embedding_list = [self.region_projector(region_embedding) for region_embedding in
                                     region_embedding_list]
        else:
            region_embedding_list = None

        mask_outputs = self.predictor(multi_scale_features, mask_features, None, seg_query, SEG_embedding,
                                      class_name_embedding, region_embedding_list)

        SEG_cls_results = mask_outputs['pred_SEG_logits']
        class_name_cls_results = mask_outputs['pred_class_name_logits']
        mask_pred_results = mask_outputs["pred_masks"]
        region_cls_results = mask_outputs['pred_region_logits']
        images = [x for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        del mask_outputs
        processed_results = []
        if SEG_cls_results is None:
            SEG_cls_results = [None]
        if class_name_cls_results is None:
            class_name_cls_results = [None]
        for _seg_info, SEG_cls_result, class_name_cls_result, mask_pred_result, input_per_image, image_size in zip(
                seg_info, SEG_cls_results, class_name_cls_results, mask_pred_results, seg_info, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            padding_mask = input_per_image.get("padding_mask")
            non_padding_indices = np.where(~ np.array(padding_mask))
            min_y, max_y = np.min(non_padding_indices[0]), np.max(non_padding_indices[0])
            min_x, max_x = np.min(non_padding_indices[1]), np.max(non_padding_indices[1])
            original_height = max_y - min_y + 1
            original_width = max_x - min_x + 1
            processed_results.append({})
            # gt = _seg_info['instances'].gt_masks
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, [original_height, original_width], height, width
                )
                # gt_result = retry_if_cuda_oom(sem_seg_postprocess)(
                #     gt, [original_height, original_width], height, width
                # )
                if SEG_cls_result is not None:
                    SEG_cls_result = SEG_cls_result.to(mask_pred_result)

            if self.semantic_on:
                semantic_r = retry_if_cuda_oom(self.class_name_semantic_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                if not self.sem_seg_postprocess_before_inference:
                    semantic_r = retry_if_cuda_oom(sem_seg_postprocess)(
                    semantic_r, [original_height, original_width], height, width
                )
                processed_results[-1]["sem_seg"] = semantic_r

            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.class_name_instance_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.class_name_panoptic_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                processed_results[-1]["panoptic_seg"] = panoptic_r
            if self.referring_on:
                instance_r = retry_if_cuda_oom(self.SEG_instance_inference)(SEG_cls_result.float(),
                                                                            mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
            if self.region_on:
                gt = _seg_info['instances'].gt_masks
                gt_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    gt, [original_height, original_width], height, width
                )
                region_cls_results = region_cls_results[0].to(mask_pred_result)
                instance_r = retry_if_cuda_oom(self.region_inference)(region_cls_results.float(),
                                                                            mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
                processed_results[-1]["gt"] = gt_result





            return processed_results
    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, past_key_values, labels, images, vp_images=None, class_name_embedding_indices=None,
            class_name_ids=None, cls_indices=None, instances=None, token_refer_id=None, refer_embedding_indices=None
    ):
        vision_tower = self.get_vision_tower()
        seg_query_mask = torch.zeros_like(input_ids)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, seg_query_mask

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        expanded_seg_query = self.seg_query.unsqueeze(0).expand(input_ids.shape[0], -1, -1)

        if (input_ids == REGION_TOKEN_INDEX).sum() != 0 and instances is not None:
            region_masks_list = [instance.vp_region_masks.tensor for instance in instances]
            vp_image_features = self.encode_images(vp_images)

            # [region_features_per_batch: [num_region, 1, dims]], len(region_features) = batch_size
            region_features = self.region_sampler(vp_image_features, region_masks_list,
                                                  original_dtype=vp_image_features.dtype,
                                                  return_dtype=vp_image_features.dtype)
            region_embedding_masks = torch.zeros_like(input_ids)
        else:
            region_features = None
            region_embedding_masks = None
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_seg_query_masks = []
        new_class_name_embedding_indices = [] if class_name_embedding_indices is not None else None
        new_refer_embedding_indices = [] if refer_embedding_indices is not None else None
        new_region_embedding_masks = [] if region_features is not None else None
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_seg_query_mask = seg_query_mask[batch_idx]
            cur_seg_query = expanded_seg_query[batch_idx]
            cur_image_feature = image_features[batch_idx]
            cur_class_name_embedding_indices = class_name_embedding_indices[batch_idx] if class_name_embedding_indices is not None else None
            cur_refer_embedding_indices = refer_embedding_indices[batch_idx] if refer_embedding_indices is not None else None
            cur_region_feature_list = region_features[batch_idx] if region_features is not None else None
            cur_region_embedding_mask = region_embedding_masks[batch_idx] if region_features is not None else None
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # ensure gradients back propagation, not changing cur_input_embeds
                cur_input_embeds = cur_input_embeds + (
                        0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                new_seg_query_masks.append(cur_seg_query_mask)
                # cur_image_idx += 1
                continue

            if labels is not None:
                cur_label = labels[batch_idx]
            else:
                cur_label = None

            if class_name_ids is not None:
                cur_class_name_ids = class_name_ids[batch_idx]
                cur_cls_indices = cls_indices[batch_idx]
            else:
                cur_class_name_ids = None
                cur_cls_indices = None
            if token_refer_id is not None:
                cur_token_refer_id = token_refer_id[batch_idx]
            else:
                cur_token_refer_id = None


            cur_class_name_embedding = self.embed_class_ids(cur_class_name_ids, cur_cls_indices)
            cur_refer_embedding = self.embed_refer_ids(cur_token_refer_id)

            cur_input_embeds, cur_label, cur_seg_query_mask, cur_class_name_embedding_indices, cur_region_embedding_mask, cur_refer_embedding_indices = self.concat_image_seg_cls_embeds(
                input_id=cur_input_ids,
                img_feature=cur_image_feature,
                label=cur_label,
                seg_query=cur_seg_query,
                seg_query_mask=cur_seg_query_mask,
                class_embed=cur_class_name_embedding,
                class_name_embedding_indices=cur_class_name_embedding_indices,
                region_embedding_mask=cur_region_embedding_mask,
                region_feature_list=cur_region_feature_list,
                refer_embedding_indices=cur_refer_embedding_indices,
                refer_embedding=cur_refer_embedding
            )
            assert cur_input_embeds.shape[0] == cur_seg_query_mask.shape[0]

            new_input_embeds.append(cur_input_embeds)
            if labels is not None:
                new_labels.append(cur_label)
            new_seg_query_masks.append(cur_seg_query_mask)
            if class_name_embedding_indices is not None:
                new_class_name_embedding_indices.append(cur_class_name_embedding_indices)
            if refer_embedding_indices is not None:
                new_refer_embedding_indices.append(cur_refer_embedding_indices)
            if new_region_embedding_masks is not None:
                new_region_embedding_masks.append(cur_region_embedding_mask)
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)),
                                          dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            new_seg_query_masks_align = []
            for new_seg_query_mask in new_seg_query_masks:
                new_seg_query_mask = torch.cat(
                    (new_seg_query_mask, torch.zeros((max_len - new_seg_query_mask.shape[0]),dtype=new_seg_query_mask.dtype, device=new_seg_query_mask.device)),
                    dim=0)
                new_seg_query_masks_align.append(new_seg_query_mask)
            new_seg_query_masks = torch.stack(new_seg_query_masks_align, dim=0)

            new_class_name_embedding_indices_align = []

            if class_name_embedding_indices is not None:
                for new_class_name_embedding_indice in new_class_name_embedding_indices:
                    new_class_name_embedding_indice = torch.cat(
                        (new_class_name_embedding_indice,
                         torch.zeros((max_len - new_class_name_embedding_indice.shape[0]),dtype=new_class_name_embedding_indice.dtype, device=new_class_name_embedding_indice.device)),
                        dim=0)
                    new_class_name_embedding_indices_align.append(new_class_name_embedding_indice)
                new_class_name_embedding_indices = torch.stack(new_class_name_embedding_indices_align, dim=0)

            if refer_embedding_indices is not None:
                new_refer_embedding_indices_align = []
                for new_refer_embedding_indice in new_refer_embedding_indices:
                    new_refer_embedding_indice = torch.cat(
                        (new_refer_embedding_indice,
                         torch.zeros((max_len - new_refer_embedding_indice.shape[0]),dtype=new_refer_embedding_indice.dtype, device=new_refer_embedding_indice.device)),
                        dim=0)
                    new_refer_embedding_indices_align.append(new_refer_embedding_indice)
                new_refer_embedding_indices = torch.stack(new_refer_embedding_indices_align, dim=0)

            if new_region_embedding_masks is not None:
                new_region_embedding_masks_align = []
                for new_region_embedding_mask in new_region_embedding_masks:
                    new_region_embedding_mask = torch.cat(
                        (new_region_embedding_mask, torch.zeros((max_len - new_region_embedding_mask.shape[0]),dtype=new_region_embedding_mask.dtype, device=new_region_embedding_mask.device)),
                        dim=0)
                    new_region_embedding_masks_align.append(new_region_embedding_mask)
                new_region_embedding_masks = torch.stack(new_region_embedding_masks_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape

        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            new_seg_query_masks = torch.stack(new_seg_query_masks, dim=0)
            if class_name_embedding_indices is not None:
                new_class_name_embedding_indices = torch.stack(new_class_name_embedding_indices, dim=0)
            if refer_embedding_indices is not None:
                new_refer_embedding_indices = torch.stack(new_refer_embedding_indices, dim=0)

            if new_region_embedding_masks is not None:
                new_region_embedding_masks = torch.stack(new_region_embedding_masks, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, new_seg_query_masks, new_class_name_embedding_indices, new_region_embedding_masks, new_refer_embedding_indices
    def eval_video(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            vp_images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            seg_info=None,
            class_name_ids=None,
            class_name_embedding_indices=None,
            cls_indices=None,
            token_refer_id=None,
            refer_embedding_indices=None,
            is_thing_list=None
    ):
        if self.panoptic_on:
            assert is_thing_list is not None, 'is_thing_list need to be given'
            self.is_thing_list = is_thing_list
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids == REGION_TOKEN_INDEX).sum() != 0:
            instances = [i['instances'] for i in seg_info]
        else:
            instances = None
        input_ids, attention_mask, past_key_values, inputs_embeds, labels, seg_query_mask, class_name_embedding_indices, region_embedding_masks, refer_embedding_indices = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images,vp_images, class_name_embedding_indices,
            class_name_ids, cls_indices, instances, token_refer_id, refer_embedding_indices)


        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs.last_hidden_state
        seg_query = self.get_seg_query(hidden_states, seg_query_mask)
        seg_query = self.seg_query_projector(seg_query)

        image_features = self.get_vision_tower_feature(images)
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features)

        if refer_embedding_indices is not None:
            SEG_embedding = self.get_SEG_embedding(hidden_states, refer_embedding_indices)
            SEG_embedding = self.SEG_token_projector(SEG_embedding)
        else:
            SEG_embedding = None

        if class_name_embedding_indices is not None:
            class_name_embedding = self.get_class_name_embedding(hidden_states, class_name_embedding_indices)
            class_name_embedding = self.class_name_projector(class_name_embedding)
        else:
            class_name_embedding = None

        if region_embedding_masks is not None:
            region_embedding_list = self.get_region_embedding(hidden_states, region_embedding_masks)
            region_embedding_list = [self.region_projector(region_embedding) for region_embedding in
                                     region_embedding_list]
        else:
            region_embedding_list = None

        mask_outputs = self.predictor(multi_scale_features, mask_features, None, seg_query, SEG_embedding,
                                      class_name_embedding, region_embedding_list)

        SEG_cls_results = mask_outputs['pred_SEG_logits']
        class_name_cls_results = mask_outputs['pred_class_name_logits']
        mask_pred_results = mask_outputs["pred_masks"]
        region_cls_results = mask_outputs['pred_region_logits']
        images = [x for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        del mask_outputs
        processed_results = []
        if SEG_cls_results is None:
            SEG_cls_results = [None]
        if class_name_cls_results is None:
            class_name_cls_results = [None]
        for _seg_info, SEG_cls_result, class_name_cls_result, mask_pred_result, input_per_image, image_size in zip(
                seg_info, SEG_cls_results, class_name_cls_results, mask_pred_results, seg_info, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            padding_mask = input_per_image.get("padding_mask")
            non_padding_indices = np.where(~ np.array(padding_mask))
            min_y, max_y = np.min(non_padding_indices[0]), np.max(non_padding_indices[0])
            min_x, max_x = np.min(non_padding_indices[1]), np.max(non_padding_indices[1])
            original_height = max_y - min_y + 1
            original_width = max_x - min_x + 1
            processed_results.append({})
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, [original_height, original_width], height, width
                )
                if SEG_cls_result is not None:
                    SEG_cls_result = SEG_cls_result.to(mask_pred_result)

            if self.semantic_on:
                semantic_r = retry_if_cuda_oom(self.class_name_semantic_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                if not self.sem_seg_postprocess_before_inference:
                    semantic_r = retry_if_cuda_oom(sem_seg_postprocess)(
                        semantic_r, [original_height, original_width], height, width
                    )
                processed_results[-1]["sem_seg"] = semantic_r

            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.class_name_instance_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.class_name_panoptic_inference)(None,
                                                                                   class_name_cls_result.float(),
                                                                                   mask_pred_result.float())
                processed_results[-1]["panoptic_seg"] = panoptic_r
            if self.referring_on:
                instance_r = retry_if_cuda_oom(self.SEG_instance_inference)(SEG_cls_result.float(),
                                                                            mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
            if self.region_on:
                gt = _seg_info['instances'].gt_masks
                gt_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    gt, [original_height, original_width], height, width
                )
                region_cls_results = region_cls_results[0].to(mask_pred_result)
                instance_r = retry_if_cuda_oom(self.region_inference)(region_cls_results.float(),
                                                                      mask_pred_result.float())
                processed_results[-1]["instances"] = instance_r
                processed_results[-1]["gt"] = gt_result

            return processed_results


AutoConfig.register("llava_phi", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, PSALMModel)
