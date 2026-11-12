#!/usr/bin/env bash
set -e

if [[ $# -lt 1 ]]; then
    echo "In order to use this container, you must bind the directory containing the images (not in subdirectories) to the container. Then, call the container, passing the bound location of the directory as the first argument. An optional second argument is the confidence threshold cutoff (defaults to 0.45).

docker run -v <path/to/images>:/DATA andersenlab/wormwatcher:latest /DATA

or

docker run -v <path/to/images>:/DATA andersenlab/wormwatcher:latest /DATA 0.65

You can also utilize GPUs by passing them as an argument to docker. This also requires that the NVIDIA Container Toolkit is installed and that your docker service is configured to use it.

docker run --gpus all -v <path/to/images>:/DATA andersenlab/wormwatcher:latest /DATA

or to use a specific device

docker run --gpus "device=0" -v <path/to/images>:/DATA andersenlab/wormwatcher:latest /DATA
"
    exit 1
fi

if [[ ! -d $1 ]]; then
    echo "Error: The image directory does not appear to exist"
    exit 1
fi

if [[ $# -gt 1 ]]; then
    threshold=$2;
else
    threshold=0.45
fi

sed "s/IMAGE_FOLDER/${1/\//\\/}/g" /WormWatcher/docker_parameters.json | sed "s/CONFIDENCE_THRESHOLD/${threshold}/g" > $1/run_parameters.json

python -u /WormWatcher/TauTracker/AnalyzePositionOnSinglePlate_nogui.py $1 $1/run_parameters.json

