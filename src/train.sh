#!/bin/bash
#SBATCH --job-name=rna_transformer
#SBATCH --account=master-bi
#SBATCH --partition=master-bi
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=rna_training_%j.log

cd /shared/home/cao/RNA-folding-predictor

MODEL_NAME=$1

singularity exec --nv /shared/home/cao/transformer_pl.sif python3 -u src/train_${MODEL_NAME}.py --output_dir "results/${MODEL_NAME}"