#!/bin/bash

#SBATCH --job-name=JOB_NAME
#SBATCH --error=wormwatcher.%j.out
#SBATCH --output=wormwatcher.%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=TIME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=a100
#SBATCH --array=ARRAY_RANGE

ml gcc/9.3.0
ml pytorch/2.5.1
eval "$(mamba shell hook --shell bash)"
mamba activate /scratch/eande106/conda_envs/wormwatcher

BATCHES=($(cat BATCH_LIST))
BATCH=${BATCHES[${SLURM_ARRAY_TASK_ID}]}

cd /scratch/eande106/WormWatcher/TauTracker
python3.9 AnalyzePositionOnSinglePlate_nogui.py ${BATCH} ${BATCH}.json
