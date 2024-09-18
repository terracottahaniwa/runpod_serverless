import base64
import io
import json
import os
import time

import runpod
import gradio as gr
from PIL import Image

import modules.scripts as scripts
from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
    Processed,
)

def clear_and_print(text: str):
    columns, liens = os.get_terminal_size()
    spaces = columns - len(text)
    print(text + " " * (spaces), end="\r")


class Script(scripts.Script):
    def title(self):
        return "RUNPOD Serverless"

    def show(self, is_img2img):
        return False if is_img2img else True

    def ui(self, is_img2img):
        RUNPOD_API_KEY = gr.Textbox(
            label="RUNPOD_API_KEY",
            value=os.getenv("RUNPOD_API_KEY"),
            type="password"
        )
        RUNPOD_ENDPOINT_ID = gr.Textbox(
            max_lines=1,
            label="RUNPOD_ENDPOINT_ID",
            value=os.getenv("RUNPOD_ENDPOINT_ID"),
        )
        return [
            RUNPOD_API_KEY,
            RUNPOD_ENDPOINT_ID,
        ]

    def run(
            self,
            p: StableDiffusionProcessing,
            RUNPOD_API_KEY,
            RUNPOD_ENDPOINT_ID,
        ):

        runpod.api_key = RUNPOD_API_KEY
        endpoint = runpod.Endpoint(RUNPOD_ENDPOINT_ID)
        is_img2img = isinstance(
            p, StableDiffusionProcessingImg2Img)

        payload = {
            "prompt": p.prompt,
            "negative_prompt": p.negative_prompt,
            "sampler": p.sampler,
            "scheduler": p.scheduler,
            "steps": p.steps,
            "width": p.width,
            "height": p.height,
            "cfg_scale": p.cfg_scale,
            "n_iter": p.n_iter,
            "batch_size": p.batch_size,
        }

        run_request = endpoint.run(
            {
                "input": payload,
            }
        )

        count = 0
        while True:
            status = run_request.status()
            clear_and_print(f"{count}: {status}")
            if status == "COMPLETED":
                print()
                break
            if status == "FAILED":
                raise Exception("FAILED")
            time.sleep(1)
            count += 1
        
        job_status = run_request._fetch_job()
        print({
            key: job_status.get(key) for key
            in ["delayTime", "executionTime"]
        })

        output: str = "".join(job_status.get("output"))
        data = json.loads(output)
        images_base64: list[str] = data.get('images', [])
        images_pil: list[Image.Image] = []
        infotexts: list[str] = []
        for image_base64 in images_base64:
            image_bytes = base64.b64decode(image_base64)
            image_bytesio = io.BytesIO(image_bytes)
            image = Image.open(image_bytesio)
            images_pil.append(image)
            infotexts.append(image.text['parameters'])
        return Processed(p, images_pil, infotexts=infotexts)