import base64
import io
import json
import os
import time

import runpod
import gradio as gr
from PIL import Image

import modules.images
import modules.shared as shared
import modules.scripts as scripts
from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
    Processed,
)
from  modules.infotext_utils import (
    parse_generation_parameters
)

BASEDIR = scripts.basedir()


def is_pil_images(images):
    return (
        isinstance(images, list)
        and len(images) != 0
        and all(
            type(image) == Image.Image
            for image in images
        )
    )


def is_b64_images(images):
    return (
        isinstance(images, list)
        and len(images) != 0
        and all(
            type(image) == str
            for image in images
        )
    )


def pil_imgs_convert_to_b64(images):
    assert is_pil_images(images)
    images_base64 = []
    for image_pil in images:
        image_bytesio = io.BytesIO()
        image_pil.save(
            image_bytesio,
            format="PNG"
        )
        image_bytes = image_bytesio.getvalue()
        image_base64 = base64.b64encode(
            image_bytes
        )
        images_base64.append(
            image_base64.decode()
        )
    return images_base64


def b64_imgs_convert_to_pil(images):
    assert is_b64_images(images)
    images_pil = []
    for image_base64 in images:
        image_bytes = base64.b64decode(image_base64)
        image_bytesio = io.BytesIO(image_bytes)
        image_pil = Image.open(image_bytesio)
        images_pil.append(image_pil)
    return images_pil


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
        WORKER = gr.Slider(
            minimum=1,
            maximum=5,
            step=1,
            label="WORKER",
        )
        return [
            RUNPOD_API_KEY,
            RUNPOD_ENDPOINT_ID,
            WORKER,
        ]

    def run(
            self,
            p: StableDiffusionProcessing,
            RUNPOD_API_KEY,
            RUNPOD_ENDPOINT_ID,
            WORKER,
        ):

        if p.scripts:
            p.scripts.before_process(p)

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

        # create payload from template
        with open(os.path.join(BASEDIR, filename)) as f:
            template = json.load(f)
            payload = {}
            for key in template.keys():
                # p has no attr "mask" but "image_mask"
                if key == "mask":
                    key = "image_mask"

                try:
                    value = getattr(p, key)
                except AttributeError:
                    continue

                if value is None:
                    continue
                elif key == "init_images":
                    convert = pil_imgs_convert_to_b64
                    payload[key] = convert(value)
                    continue
                elif key == "image_mask":
                    # payload need "mask" not "image_mask"
                    convert = pil_imgs_convert_to_b64
                    payload["mask"] = convert([value])[0]
                    continue
                elif key == "script_args":
                    # not support yet
                    continue
                else:
                    payload[key] = value

        run_requests = []
        for i in range(WORKER):
            request_input = {
                "input": {
                    "is_img2img": is_img2img,
                    "payload": payload,
                }
            }
            run_request = endpoint.run(request_input)
            run_requests.append(run_request)

        count = 0
        count_up = 1
        for i, run_request in enumerate(run_requests):
            while True:
                status = run_request.status()
                shared.state.textinfo = f"worker{i}: {count}s {status}"
                if status == "COMPLETED":
                    break
                if status == "FAILED":
                    for run_request in run_requests:
                        run_request.cancel()
                    raise Exception("FAILED")
                if status == "CANCELLED":
                    raise Exception("CANCELLED")
                if (
                    shared.state.interrupted
                    or shared.state.stopping_generation
                ):
                    for run_request in run_requests:
                        run_request.cancel()
                time.sleep(count_up)
                count += count_up   

        all_img = []
        all_txt = []
        for j, run_request in enumerate(run_requests):
            shared.state.textinfo = f"fetch job{j}"
            job_status = run_request._fetch_job()
            job_times = {
                key: job_status.get(key) for key
                in ["delayTime", "executionTime"]
            }
            delay_time, execution_time = job_times.values()
            print(f"worker{j}: {job_times}")

            output: str = "".join(job_status.get("output"))
            data = json.loads(output)
            images_base64 = data.get("images", [])
            images_pil = b64_imgs_convert_to_pil(images_base64)
            infotexts = []
            for i, image_pil in enumerate(images_pil):
                shared.state.textinfo = f"save image: {i}"
                infotext = image_pil.text.get("parameters")
                infotexts.append(infotext)
                info = parse_generation_parameters(infotext)
                is_save_sample = p.save_samples()
                if is_save_sample:
                    modules.images.save_image(
                        image=image_pil,
                        path=p.outpath_samples,
                        basename="runpod",
                        seed=info.get("Seed"),
                        prompt=info.get("Prompt"),
                        extension=shared.opts.samples_format,
                        info=infotext,
                        p=p
                    )
            infotexts = [
                infotext
                + f", delayTime: {delay_time / 1000:.2f}s, "
                + f"executionTime: {execution_time / 1000:.2f}s"
                for infotext in infotexts
            ]
            all_img += images_pil
            all_txt += infotexts
        
        return Processed(p, all_img, infotexts=all_txt)
