FROM ubuntu:jammy

RUN export DEBIAN_FRONTENF=noninteractive; \
apt update; apt install -y libgl1-mesa-glx libgtk2.0-0 libtcmalloc-minimal4 build-essential python3-pip python3-venv bc wget curl git; \
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

WORKDIR /stable-diffusion-webui
RUN pip3 install httpx runpod; \
./webui.sh -f --no-download-sd-model --no-hashing --skip-version-check --skip-python-version-check --skip-torch-cuda-test --xformers --exit; \
wget --content-disposition -P models/VAE-approx/ "https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/download/v1.0.0-pre/model.pt"; \
wget --content-disposition -P models/VAE-approx/ "https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/download/v1.0.0-pre/vaeapprox-sd3.pt"; \
wget --content-disposition -P models/VAE-approx/ "https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/download/v1.0.0-pre/vaeapprox-sdxl.pt"; \
. venv/bin/activate; python -c "import huggingface_hub; huggingface_hub.snapshot_download('openai/clip-vit-large-patch14', allow_patterns=['*.txt', '*.json'])"; \
apt autoremove -y; rm -rf /var/cache/apt/; rm -rf /root/.cache/pip/

COPY models/Lora/ models/Lora/
COPY models/Stable-diffusion/ models/Stable-diffusion/

COPY handler.py .
COPY start.sh .

CMD ["bash", "start.sh"]