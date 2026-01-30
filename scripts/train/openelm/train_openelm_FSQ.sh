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

bash /scratch/drjieliu_root/drjieliu/gjiajun/TinyLLaVA_Factory/scripts/train/openelm/pretrain_openelm_FSQ.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$VQ_VERSION" "$VQ_SIZE" "$DATE"
bash /scratch/drjieliu_root/drjieliu/gjiajun/TinyLLaVA_Factory/scripts/train/openelm/finetune_openelm_FSQ.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$VQ_VERSION" "$VQ_SIZE" "$DATE"