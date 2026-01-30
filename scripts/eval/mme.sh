#!/bin/bash

MODEL_PATH="/scratch/drjieliu_root/drjieliu/gjiajun/output/tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-finetune-Ql-size-8-Dec_27/"
MODEL_NAME="tiny-llava-OpenELM-270M-Instruct-siglip-so400m-patch14-384-elm_base-finetune-Ql-size-8-Dec_27"
EVAL_DIR="/scratch/drjieliu_root/drjieliu/gjiajun/dataset/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/MME/llava_mme.jsonl \
    --image-folder $EVAL_DIR/MME/MME_Benchmark_release_version \
    --answers-file $EVAL_DIR/MME/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
   --conv-mode llama

cd $EVAL_DIR/MME

python convert_answer_to_mme.py --experiment $MODEL_NAME

cd eval_tool

python calculation.py --results_dir answers/$MODEL_NAME

