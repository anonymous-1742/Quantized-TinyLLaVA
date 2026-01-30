#!/bin/bash
#SBATCH --job-name=tinyllava_split (TODO)
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=24
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/drjieliu_root/drjieliu/gjiajun/output_log/%x-%j.log (TODO)
#SBATCH --error=/scratch/drjieliu_root/drjieliu/gjiajun/logs/%x-%j-E.log (TODO)
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zjiayin@umich.edu (TODO)
echo "=================================="
echo "Starting to tile svs files"
echo "=================================="

conda activate tinyllava_factory
cd /scratch/drjieliu_root/drjieliu/gjiajun/TinyLLaVA_Factory
bash split_learning/train_openelm_FSQ-split_learning.sh

echo "=================================="
echo "svs file tiled completed!"
echo "=================================="