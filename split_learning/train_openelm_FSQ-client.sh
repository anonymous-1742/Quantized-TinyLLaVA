DATA_PATH=/nfs/turbo/umms-drjieliu1/usr/gjiajun/dataset/text_files/blip_laion_cc_sbu_558k.json #pretrain annotation file path
FINETUNE_DATA_PATH=/nfs/turbo/umms-drjieliu1/usr/gjiajun/dataset/text_files/llava_v1_5_mix665k.json #finetune annotation file path
IMAGE_PATH=/nfs/turbo/umms-drjieliu1/usr/gjiajun/dataset/llava/llava_pretrain/images #pretrain image dir
FINETUNE_IMAGE_PATH=/nfs/turbo/umms-drjieliu1/usr/gjiajun/dataset #finetune image dir


LLM_VERSION=apple/OpenELM-270M-Instruct
VT_VERSION=google/siglip-so400m-patch14-384
VT_VERSION2=""
CN_VERSION=mlp2x_gelu
VQ_VERSION="FSQ"
VQ_SIZE=${1:-4}
CONV_VERSION=llama
VERSION=elm_base
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=2048
DATE="Nov_3_2epoch"
PORT=${2:-1111}
trap 'echo "[Main] Exiting, killing all child processes"; pkill -P $$; exit 1' SIGINT SIGTERM



PORT=${PORT} CUDA_VISIBLE_DEVICES=0 bash /nfs/turbo/umms-drjieliu1/usr/gjiajun/TinyLLaVA_Factory/split_learning/pretrain_openelm_FSQ-client.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$VQ_VERSION" "$VQ_SIZE" "$DATE" &
CLIENT_PID=$!

wait $CLIENT_PID


#CUDA_VISIBLE_DEVICES=0 bash /nfs/turbo/umms-drjieliu1/usr/gjiajun/TinyLLaVA_Factory/split_learning/finetune_openelm_FSQ-client.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$VQ_VERSION" "$VQ_SIZE" "$DATE" &
#CLIENT_PID=$!

#wait $CLIENT_PID
