FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --reinstall -y --no-install-recommends \
    git \
    procps \
    tini \
    libxml2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mamba update && \
    mamba install -y \
    numpy \
    scipy \
    pandas \
    opencv \
    matplotlib \
    xlsxwriter&& \
    mamba clean -afy

RUN pip install --no-cache-dir ultralytics

RUN git clone https://github.com/AndersenLab/2026_counting_broodsize_images_YOLO.git && \
    mv 2026_counting_broodsize_images_YOLO/WormWatcher /WormWatcher && \
    rm -rf 2026_counting_broodsize_images_YOLO

COPY --chmod=777 entrypoint.sh /usr/local/bin/

COPY --chmod=444 yolo8_2026-02-16_detectmode.pt /WormWatcher/NeuralNets/

ENTRYPOINT [ "tini", "--", "/usr/local/bin/entrypoint.sh" ]
