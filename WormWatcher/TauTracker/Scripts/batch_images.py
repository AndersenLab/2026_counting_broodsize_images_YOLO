#!/usr/bin/env python

import argparse
import glob
import math
import os
import random
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Split files into batches and create parameters files",
        epilog="""
Example usage:
    python batch_images.py /path/to/image/folder /path/to/batch/prefix --batch-size 500
        """
    )
    parser.add_argument('image_folder', help='Image folder to process')
    parser.add_argument('batch_prefix', help='Path and prefix for output batches')
    parser.add_argument('--batch-size', type=int, default=500, help='Number of images per batch')
    parser.add_argument('--json-template', type=str, default='./parameters_template.json', help='Parameters template file')
    parser.add_argument('--slurm-template', type=str, default='./slurm_template.sh', help='Slurm template file')
    parser.add_argument('--confidence-threshold', type=float, default=0.45, help='Object confidence threshold')

    args = parser.parse_args()

    # Get image names
    file_names = glob.glob(f"{args.image_folder}/*.png") + glob.glob(f"{args.image_folder}/*.PNG")

    # Randomize image names
    random.shuffle(file_names)

    # Load parameters template
    json_template = open(args.json_template).read()

    # Determine how many batches are needed
    just = int(math.log10(len(file_names)))
    num_batches = (len(file_names) - 1) // args.batch_size + 1

    batch_names = []
    # Create batches
    for i in range(num_batches):
        start = i * args.batch_size
        stop = min(len(file_names), start + args.batch_size)

        # Create batch folder
        suffix = f'{i}'.rjust(just, '0')
        batch_name = f"{os.path.abspath(args.batch_prefix)}_{suffix}"
        batch_names.append(batch_name)
        os.mkdir(batch_name)

        # Populate batch folder
        for j in range(start, stop):
            file_path = os.path.abspath(file_names[j])
            file_name = file_path.split('/')[-1]
            os.symlink(file_path, f"{batch_name}/{file_name}")

        # Create batch parameter json
        batch_json = str(json_template)
        batch_json = batch_json.replace('IMAGE_FOLDER', batch_name)
        batch_json = batch_json.replace('CONFIDENCE_THRESHOLD', f"{args.confidence_threshold}")
        fs = open(f"{batch_name}.json", 'w')
        fs.write(batch_json)
        fs.close()

    # create batch list
    fs = open(f"{os.path.abspath(args.batch_prefix)}.batchlist.txt", 'w')
    for name in batch_names:
        fs.write(f"{name}\n")
    fs.close()

    # Create slurm submission script
    slurm_script = open(args.slurm_template).read()
    slurm_script = slurm_script.replace("JOB_NAME", f"{args.batch_prefix.split('/')[-1]}")
    slurm_script = slurm_script.replace("ARRAY_RANGE", f"0-{num_batches - 1}%8")
    slurm_script = slurm_script.replace("BATCH_LIST", f"{os.path.abspath(args.batch_prefix)}.batchlist.txt")
    slurm_script = slurm_script.replace("TIME", f"{int(math.ceil(args.batch_size / 250))}")
    fs = open(f"{os.path.abspath(args.batch_prefix)}.slurm.sh", 'w')
    fs.write(slurm_script)
    fs.close()


main()
