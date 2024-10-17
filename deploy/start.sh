#!/bin/bash
./webui.sh -f --skip-version-check --skip-python-version-check --skip-torch-cuda-test --no-download-sd-model --no-hashing --skip-install --xformers --no-half-vae --api --nowebui &
python3 handler.py
