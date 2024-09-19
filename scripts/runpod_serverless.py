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

BASEDIR = scripts.basedir()


def clear_and_print(text: str):
    columns, lines = os.get_terminal_size()
    spaces = columns - len(text)
    print(text + " " * (spaces), end="\r")


def is_pil_images(images):
    return (
        (True if images else False)
        and isinstance(images, list)
        and all(
            type(image) == Image.Image
            for image in images
        )
    )


def pil_imgs_convert_to_b64(images):
    assert is_pil_images(images)
    images_base64 = []
    for image_pil in images:
        image_bytesio = io.BytesIO()
        image_pil.save(image_bytesio,
                        format="PNG")
        image_bytes = image_bytesio.getvalue()
        image_base64 = base64.b64encode(
            image_bytes
        )
        images_base64.append(
            image_base64.decode()
        )
    return images_base64


def b64_img_convert_to_pil(image_base64):
    assert isinstance(image_base64, str)
    image_bytes = base64.b64decode(image_base64)
    image_bytesio = io.BytesIO(image_bytes)
    image_pil = Image.open(image_bytesio)
    return image_pil


class Script(scripts.Script):
    def title(self):
        return "RUNPOD Serverless"

    def show(self, is_img2img):
        return True

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
            p,
            StableDiffusionProcessingImg2Img
        )

        if is_img2img:
            filename = "img2img.json"
        else:
            filename = "txt2img.json"
        with open(os.path.join(BASEDIR, filename)) as f:
            template = json.load(f)
            payload = {}
            for key in template.keys():
                try:
                    value = getattr(p, key)
                except AttributeError:
                    continue
                if value:
                    converter = pil_imgs_convert_to_b64
                    if is_pil_images(value):
                        value = converter(value)
                    payload[key] = value

        run_request = endpoint.run(
            {
                "input": {
                    "is_img2img": is_img2img,
                    "payload": payload,
                }
            }
        )

        count = 0
        while True:
            status = run_request.status()
            clear_and_print(f"{count}: {status}")
            if status == "COMPLETED":
                print()
                break
            if status == "CANCELLED":
                raise Exception("CANCELLED")
            if status == "FAILED":
                raise Exception("FAILED")
            time.sleep(1)
            count += 1
        
        job_status = run_request._fetch_job()
        print(
            {
                key: job_status.get(key) for key
                in ["delayTime", "executionTime"]
            }
        )

        output: str = "".join(job_status.get("output"))
        data = json.loads(output)
        images_base64: list[str] = data.get('images', [])
        images_pil: list[Image.Image] = []
        infotexts: list[str] = []
        for image_base64 in images_base64:
            image_pil = b64_img_convert_to_pil(image_base64)
            images_pil.append(image_pil)
            infotexts.append(image_pil.text['parameters'])
        return Processed(p, images_pil, infotexts=infotexts)
