#!/bin/bash

#SBATCH --job-name=JOB_NAME
#SBATCH --error=wormwatcher.%j.out
#SBATCH --output=wormwatcher.%j.out
#SBATCH --time=TIME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=ARRAY_RANGE

ml pytorch/2.4.1
eval "$(mamba shell hook --shell bash)"
mamba activate /data/eande106/software/conda_envs/wormwatcher

BATCHES=($(cat BATCH_LIST))
BATCH=${BATCHES[${SLURM_ARRAY_TASK_ID}]}

cd /data/eande106/software/2026_counting_broodsize_images_YOLO/WormWatcher/TauTracker
python3.9 AnalyzePositionOnSinglePlate_nogui.py ${BATCH} ${BATCH}.json

