#!/bin/bash
torchrun --nproc_per_node=4 \
         --master_port=29500 \
         /home/jiangtianyuan/resource/voice/vocalnet/view_model/view2.py \
         --use_ddp \
         --s2s \
         --query_audio ./omni_speech/infer/llama_questions_42.wav \
         --save_dir ./generated_audio