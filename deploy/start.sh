#!/bin/bash
./webui.sh -f --no-download-sd-model --no-hashing --skip-version-check --skip-python-version-check --skip-torch-cuda-test --skip-install --xformers --opt-channelslast --disable-nan-check --no-half-vae --api --nowebui &

while true; do
  if ss -l | grep 7861 ; then
    break
  fi
  sleep 1
done

python3 handler.py
