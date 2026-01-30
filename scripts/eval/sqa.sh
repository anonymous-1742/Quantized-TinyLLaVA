#!/bin/bash

MODEL_PATH="/scratch/drjieliu_root/drjieliu/gjiajun/output/tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-finetune-FSQ-size4/"
MODEL_NAME="tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-finetune-FSQ-size4"
EVAL_DIR="/scratch/drjieliu_root/drjieliu/gjiajun/dataset/eval"

python -m tinyllava.eval.model_vqa_science \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/scienceqa/llava_test_CQM-A.json \
    --image-folder $EVAL_DIR/scienceqa/images/test \
    --answers-file $EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llama

python tinyllava/eval/eval_science_qa.py \
    --base-dir $EVAL_DIR/scienceqa \
    --result-file $EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl \
    --output-file $EVAL_DIR/scienceqa/answers/"$MODEL_NAME"_output.jsonl \
    --output-result $EVAL_DIR/scienceqa/answers/"$MODEL_NAME"_result.json

