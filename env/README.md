In order to use this container, you must bind the directory containing the images (not in subdirectories) to the container. Then, call the container, passing the bound location of the directory as the first argument. An optional second argument is the confidence threshold cutoff (defaults to 0.45).

```bash
docker run -v <path/to/images>:/DATA andersenlab/wormwatcher:latest /DATA
```

or

```bash
docker run -v <path/to/images>:/DATA andersenlab/wormwatcher:latest /DATA 0.65
```

You can also utilize GPUs by passing them as an argument to docker. This also requires that the NVIDIA Container Toolkit is installed and that your docker service is configured to use it.

```bash
docker run --gpus all -v <path/to/images>:/DATA andersenlab/wormwatcher:latest /DATA
```

or to use a specific device

```bash
docker run --gpus "device=0" -v <path/to/images>:/DATA andersenlab/wormwatcher:latest /DATA
```

If you want to be able to terminate the container early, you can use `docker stop` from a different terminal or run the container in interactive more using the docker flags `-it`. This allows you to press `ctrl-c` to cancel the run.