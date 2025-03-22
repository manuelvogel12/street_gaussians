#!/bin/bash
#SBATCH --job-name="Street gaussian rendering"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:12
#SBATCH --mem=64G
#SBATCH --exclude=node1,node7
#SBATCH --time=0:30:00
#SBATCH --mail-type=END
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

# srun --unbuffered python3 train.py --job_name "$SLURM_JOB_NAME" --job_id "$SLURM_JOB_ID"
srun --unbuffered /home/stud/vogm/miniconda3/envs/ma/bin/python render.py --config configs/waymo_train_carla_0019_lidar.yaml mode lidar
