#!/bin/bash

export WANDB_MODE="offline"
export TOKENIZERS_PARALLELISM=false

CHECKPOINT_NAME="llama32-1B-instruct-s2s-mtp"
CHECKPOINT_DIR="./checkpoints/${CHECKPOINT_NAME}"
BASE_MODEL="./checkpoints/llama32-1B-instruct-s2t" 
DATA_PATH="./playground/VoiceAssitant-430K-VocalNet/VoiceAssistant-430K.json"
SPEECH_FOLDER="./playground/"
SPEECH_ENCODER="./models/speech_encoder/whisper-large-v3"

# 创建日志目录
mkdir -p "$CHECKPOINT_DIR"

# 获取当前日期和时间
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${CHECKPOINT_DIR}/${CHECKPOINT_NAME}-${timestamp}.log"
echo "Starting training at $(date)"
echo "Logging to $log_file"


deepspeed omni_speech/train/train_mem.py \
    --lora_enable False \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $BASE_MODEL \
    --version llama_3 \
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
    --num_train_epochs 3 \
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


