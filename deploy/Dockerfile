FROM ubuntu:jammy

RUN export DEBIAN_FRONTENF=noninteractive; \
apt update; apt install -y libgl1-mesa-glx libgtk2.0-0 libtcmalloc-minimal4 build-essential python3-pip python3-venv bc wget curl git; \
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git; \
apt autoremove -y; rm -rf /var/cache/apt/

WORKDIR /stable-diffusion-webui
RUN ./webui.sh -f --skip-version-check --skip-python-version-check --skip-torch-cuda-test --no-download-sd-model --no-hashing --xformers --exit
RUN wget --content-disposition -P models/Stable-diffusion "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true"
RUN wget --content-disposition -P models/VAE-approx/ "https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/download/v1.0.0-pre/vaeapprox-sdxl.pt"
RUN . venv/bin/activate; python -c "import huggingface_hub; huggingface_hub.snapshot_download('openai/clip-vit-large-patch14', allow_patterns=['*.txt', '*.json'])"

RUN pip3 install httpx runpod; rm -rf /root/.cache/pip/
COPY handler.py .
COPY start.sh .

CMD ["bash", "start.sh"]