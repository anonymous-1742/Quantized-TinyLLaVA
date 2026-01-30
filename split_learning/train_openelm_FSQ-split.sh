DATA_PATH=/scratch/drjieliu_root/drjieliu/gjiajun/dataset/text_files/blip_laion_cc_sbu_558k.json #pretrain annotation file path
FINETUNE_DATA_PATH=/scratch/drjieliu_root/drjieliu/gjiajun/dataset/text_files/llava_v1_5_mix665k.json #finetune annotation file path
IMAGE_PATH=/scratch/drjieliu_root/drjieliu/gjiajun/dataset/llava/llava_pretrain/images #pretrain image dir
FINETUNE_IMAGE_PATH=/scratch/drjieliu_root/drjieliu/gjiajun/dataset #finetune image dir


LLM_VERSION=apple/OpenELM-270M-Instruct
VT_VERSION=google/siglip-so400m-patch14-384
VT_VERSION2=""
CN_VERSION=mlp2x_gelu
VQ_VERSION="FSQ"
VQ_SIZE=4
CONV_VERSION=llama
VERSION=elm_base
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=2048
DATE="Nov_3_2epoch"

CUDA_VISIBLE_DEVICES=0 bash /scratch/drjieliu_root/drjieliu/gjiajun/TinyLLaVA_Factory/split_learning/pretrain_openelm_FSQ-server.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$VQ_VERSION" "$VQ_SIZE" "$DATE" &
SERVER_PID=$!

while ! lsof -i :9999 >/dev/null 2>&1; do
    sleep 1
done

echo "[Main] Server ready"

CUDA_VISIBLE_DEVICES=0 bash /scratch/drjieliu_root/drjieliu/gjiajun/TinyLLaVA_Factory/split_learning/pretrain_openelm_FSQ-client.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$VQ_VERSION" "$VQ_SIZE" "$DATE" &
CLIENT_PID=$!

wait $SERVER_PID
wait $CLIENT_PID

bash /scratch/drjieliu_root/drjieliu/gjiajun/TinyLLaVA_Factory/split_learning/finetune_openelm_FSQ-server.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$VQ_VERSION" "$VQ_SIZE" "$DATE" &
SERVER_PID=$!

while ! lsof -i :9999 >/dev/null 2>&1; do
    sleep 1
done

echo "[Main] Server ready"

bash /scratch/drjieliu_root/drjieliu/gjiajun/TinyLLaVA_Factory/split_learning/finetune_openelm_FSQ-client.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$VQ_VERSION" "$VQ_SIZE" "$DATE" &
CLIENT_PID=$!

wait $SERVER_PID
wait $CLIENT_PID