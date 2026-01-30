#!/bin/bash
export PYTHONPATH="/nfs/turbo/umms-drjieliu1/usr/gjiajun/TinyLLaVA_Factory:$PYTHONPATH"
if [ $# -ne 12 ]; then
    echo "Usage: $0 <DATA_PATH> <IMAGE_PATH> <LLM_VERSION> <VT_VERSION> <VT_VERSION2> <CN_VERSION> <VERSION> <TRAIN_RECIPE> <MODEL_MAX_LENGTH> <VQ_VERSION> <VQ_SIZE> <DATE>"
    exit 1
fi

# Assign the arguments to variables
DATA_PATH="$1"
IMAGE_PATH="$2"
LLM_VERSION="$3"
VT_VERSION="$4"
VT_VERSION2="$5"
CN_VERSION="$6"
VERSION="$7"
TRAIN_RECIPE="$8"
MODEL_MAX_LENGTH="$9"
VQ_VERSION="${10}"
VQ_SIZE="${11}"
VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"
DATE="${12}"


#deepspeed --include localhost:0 --master_port 29503 tinyllava/train/tinyllava_client.py \
#    --deepspeed ./scripts/zero3.json \
python tinyllava/train/tinyllava_client.py \
    --data_path  $DATA_PATH\
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version pretrain \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --vq_type $VQ_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --fp16 False \
    --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm frozen \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --output_dir /nfs/turbo/umms-drjieliu1/usr/gjiajun/pretrain_client/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-client-pretrain-${VQ_VERSION}-size-${VQ_SIZE}-${DATE} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 8700 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --client_layer_num 5\
    --run_name tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-pretrain\
    --train_VQ codebook_frozen\
    --discrete_size $VQ_SIZE
