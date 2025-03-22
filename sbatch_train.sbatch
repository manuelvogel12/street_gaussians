#!/bin/bash
#SBATCH --job-name="Street Gaussian Training C17"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=60G
#SBATCH --exclude=node1,node7
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

# srun --unbuffered python3 train.py --job_name "$SLURM_JOB_NAME" --job_id "$SLURM_JOB_ID"

eval "$(conda shell.bash hook)"  # Ensure Conda is properly initialized
conda activate ma


srun --unbuffered /home/stud/vogm/miniconda3/envs/ma/bin/python train.py --config configs/waymo_train_carla_0019.yaml && \
srun --unbuffered /home/stud/vogm/miniconda3/envs/ma/bin/python render.py --config configs/waymo_train_carla_0019.yaml mode trajectory
