#!/bin/bash

MODEL_PATH="/scratch/drjieliu_root/drjieliu/gjiajun/output/tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-finetune-Ql-size-8-Dec_27/"
MODEL_NAME="tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-finetune-Ql-size-8-Dec_27"
EVAL_DIR="/scratch/drjieliu_root/drjieliu/gjiajun/dataset/eval"


python -m tinyllava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/mm-vet/llava-mm-vet.jsonl \
    --image-folder $EVAL_DIR/mm-vet/images \
    --answers-file $EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode llama

mkdir -p $EVAL_DIR/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl \
    --dst $EVAL_DIR/mm-vet/results/$MODEL_NAME.json
