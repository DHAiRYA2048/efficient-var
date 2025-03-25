#!/bin/bash
#SBATCH --job-name=hart_fid_baseline
#SBATCH --output=hart_inference_%j.out
#SBATCH --error=hart_inference_%j.err
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=pr_169_general

# Load required modules (update these as needed)
module load cuda/11.7
module load python/3.8

# Activate your virtual environment if needed
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate ../hart_venv/

# Run the inference script with the JSONL file as input
python 5k_inference.py --jsonl_file common_patch/5k_pairs.jsonl

