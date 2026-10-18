#!/bin/bash

SCRIPT="download_dataset"
# export HF_ENDPOINT=https://hf-mirror.com
while true; do
    # huggingface-cli download --repo-type dataset robbyant/robotwin-clean-and-aug-lerobot --local-dir "${ROBOTWIN_DOWNLOAD_DIR:-./robotwin-clean-and-aug-lerobot}"
    HF_ENDPOINT=https://hf-mirror.com python -m ${SCRIPT}
    if [ $? -eq 0 ]; then
        echo "Download succeeded."
        break
    else
        echo "Download failed. Retrying..."
        sleep 2
    fi
done