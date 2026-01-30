#!/bin/bash

MODEL_PATH="/scratch/drjieliu_root/drjieliu/gjiajun/output/tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-finetune-Ql-size-8-Dec_27/"
MODEL_NAME="tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-finetune-Ql-size-8-Dec_27"
EVAL_DIR="/scratch/drjieliu_root/drjieliu/gjiajun/dataset/eval"

python -m tinyllava.eval.model_vqa_pope \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode llama

python tinyllava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl
