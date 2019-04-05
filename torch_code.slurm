#!/bin/bash
#
#SBATCH --job-name=torch-gpu
#SBATCH --output=res.txt
#SBATCH --ntasks=28
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH -p gpu-long
#SBATCH -o torch_gpu.out
#SBATCH -e torch_gpu.err

module load shared
module load cuda91/toolkit/9.1
module load cudnn/6.0

source /gpfs/scratch/hsarkar/anaconda3/etc/profile.d/conda.sh
echo "activating conda"
conda deactivate
conda activate torch-base 

cd /gpfs/scratch/hsarkar/attention_mechanism/Attention4DNASeqRepresentation

python AttentionModelTorch.py