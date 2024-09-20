# RUNPOD severless
## A extension of stable diffusion webui for generate images on the runpod serverless

## deploy
```
cd ~/stable-diffusion-webui/extensions/runpod_serverless/deploy
cp YOUR_MODEL_PATH models/Stable-diffusion
cp YOUR_LORA_PATH models/Lora
docker build -t YOUR_DOCKER_NAME/REPO_NAME:TAG .
docker push YOUR_DOCKER_NAME/REPO_NAME:TAG

# and deploy runpod endpoint with the docker image.
```

## install extension
```
cd ~/stable-diffusion-webui/extensions
git clone git@github.com:terracottahaniwa/runpod_serverless.git

# or install from extensions tab in webui.
```

## usage
select RUNPOD Serverless from scripts in the webui.  
input your runpod api key and endpoint id, then generate images.  