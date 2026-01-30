#!/bin/bash

MODEL_PATH="/scratch/drjieliu_root/drjieliu/gjiajun/output/tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-finetune-Ql-size-8-Dec_27/"
MODEL_NAME="tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-finetune-Ql-size-8-Dec_27"
EVAL_DIR="/scratch/drjieliu_root/drjieliu/gjiajun/dataset/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $EVAL_DIR/textvqa/train_images \
    --answers-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode llama

python -m tinyllava.eval.eval_textvqa \
    --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val.json \
    --result-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl

