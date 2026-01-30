DATA_PATH=/nfs/turbo/umms-drjieliu1/usr/gjiajun/dataset/text_files/blip_laion_cc_sbu_558k.json #pretrain annotation file path
FINETUNE_DATA_PATH=/nfs/turbo/umms-drjieliu1/usr/gjiajun/dataset/text_files/llava_v1_5_mix665k.json #finetune annotation file path
IMAGE_PATH=/nfs/turbo/umms-drjieliu1/usr/gjiajun/dataset/llava/llava_pretrain/images #pretrain image dir
FINETUNE_IMAGE_PATH=/nfs/turbo/umms-drjieliu1/usr/gjiajun/gjiajun/dataset #finetune image dir


LLM_VERSION=apple/OpenELM-270M-Instruct
VT_VERSION=google/siglip-so400m-patch14-384
VT_VERSION2=""
CN_VERSION=mlp2x_gelu
VQ_VERSION="None"
VQ_SIZE=${1:-4}
CONV_VERSION=llama
VERSION=elm_base
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=2048
DATE="Nov_3_2epoch"
PORT=${2:-1111}
NAME=${3:-"Jiajun"}

trap 'echo "[Main] Exiting, killing all child processes"; pkill -P $$; exit 1' SIGINT SIGTERM

LOG_DIR=/nfs/turbo/umms-drjieliu1/usr/gjiajun/TinyLLaVA_Factory/logs
mkdir -p ${LOG_DIR}

PRETRAIN_LOG=${LOG_DIR}/server_pretrain_${VQ_VERSION}_size_${VQ_SIZE}_$NAME.log

PORT=${PORT} PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
bash /nfs/turbo/umms-drjieliu1/usr/gjiajun/TinyLLaVA_Factory/split_learning/pretrain_openelm_FSQ-server.sh \
"$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" \
"$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$VQ_VERSION" "$VQ_SIZE" "$DATE" \
2>&1 | tee "${PRETRAIN_LOG}" &

SERVER_PID=$!

while ! lsof -i :9999 >/dev/null 2>&1; do
    sleep 1
done

echo "[Main] Pretrain server ready"

wait $SERVER_PID

FINETUNE_LOG=${LOG_DIR}/server_finetune_${VQ_VERSION}.log

#PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
#bash /nfs/turbo/umms-drjieliu1/usr/gjiajun/TinyLLaVA_Factory/split_learning/finetune_openelm_FSQ-server.sh \
#"$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" \
#"$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" \
#"$VQ_VERSION" "$VQ_SIZE" "$DATE" \
#2>&1 | tee "${FINETUNE_LOG}" &

#SERVER_PID=$!

#while ! lsof -i :9999 >/dev/null 2>&1; do
    sleep 1
#done

#echo "[Main] Finetune server ready"

#wait $SERVER_PID