################## VICUNA ##################
# PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

################## opt ##################
# PROMPT_VERSION="opt"
# MODEL_VERSION="opt-iml-1.3b"
################## LLaMA-2 ##################
export DISABLE_ADDMM_CUDA_LT=1
deepspeed psalm/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "/mnt/output/llava_sam/llava_phi_swin_v1_5_2e-5" \
    --version "llava_phi" \
    --instance_json_path "./datasets/lisa/instruction_segmentation_train.json" \
    --region_json_path "./datasets/lisa/visual_prompt_segmentation_train.json" \
    --panoptic_json_path "/mnt/output/pretrained_models/panoptic_coco_zem" \
    --ref_coco_path "/mnt/output/pretrained_models/llm_seg/llava_swin_seg/data/refcoco/refcoco_train.json" \
    --ref_coco_plus_path "/mnt/output/pretrained_models/llm_seg/llava_swin_seg/data/refcoco+/refcoco+_train.json" \
    --ref_coco_g_path "/mnt/output/pretrained_models/llm_seg/llava_swin_seg/data/refcocog/refcocog_train.json" \
    --refcoco_image_folder "./datasets/lisa/refer_seg/images/mscoco/images/train2014" \
    --image_folder "./datasets/lisa/coco/train2017" \
    --mmconv_path "/mnt/output/pretrained_models/mm_data_zem" \
    --vision_tower "/mnt/output/pretrained_models/model_final_pan_base.pkl" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 False \
    --output_dir $1 \
    --num_train_epochs $4 \
    --per_device_train_batch_size $2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $5 \
    --save_total_limit 1 \
    --learning_rate $3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb