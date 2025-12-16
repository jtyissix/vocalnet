#!/bin/bash

export WANDB_MODE="offline"
export TOKENIZERS_PARALLELISM=false

CHECKPOINT_NAME="qwen25-7b-s2s-mtp"
CHECKPOINT_DIR="./checkpoints/${CHECKPOINT_NAME}"
BASE_MODEL="" 
DATA_PATH=""
SPEECH_FOLDER=""
SPEECH_ENCODER="./models/speech_encoder/whisper-large-v3"

# 创建日志目录
mkdir -p "$CHECKPOINT_DIR"

# 获取当前日期和时间
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${CHECKPOINT_DIR}/${CHECKPOINT_NAME}-${timestamp}.log"
echo "Starting training at $(date)"
echo "Logging to $log_file"

deepspeed --master_port 29602 --include localhost:0,1,2,3 omni_speech/train/train_mem.py \
    --lora_enable False \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $BASE_MODEL \
    --version qwen_2_5 \
    --input_type mel \
    --mel_size 128 \
    --speech_encoder_type whisper \
    --speech_projector_type linear \
    --tune_speech_generator_only True \
    --has_speech_generator True \
    --mm_tunable_parts "speech_generator" \
    --is_multimodal \
    --data_path $DATA_PATH \
    --speech_folder $SPEECH_FOLDER \
    --speech_encoder $SPEECH_ENCODER \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$CHECKPOINT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb &> "$log_file"


