import json
import os
import time
from threading import Thread

import runpod
import gradio as gr

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

from scripts.utils import (
    is_pil_images,
    is_b64_images,
    pil_imgs_convert_to_b64,
    b64_imgs_convert_to_pil,
    Timer,
    ReturnableThread,
)

BASEDIR = scripts.basedir()


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
        WORKERS = gr.Slider(
            minimum=1,
            maximum=5,
            step=1,
            label="WORKERS",
        )
        return [
            RUNPOD_API_KEY,
            RUNPOD_ENDPOINT_ID,
            WORKERS,
        ]

    def run(
            self,
            p: StableDiffusionProcessing,
            RUNPOD_API_KEY,
            RUNPOD_ENDPOINT_ID,
            WORKERS,
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
            template = "img2img.json"
        else:
            template = "txt2img.json"

        # create payload from template
        path = os.path.join(BASEDIR, template)
        with open(path) as f:
            template = json.load(f)
            payload = {}

            for key in template.keys():
                convert = pil_imgs_convert_to_b64

                # p has no attr "mask" but "image_mask"
                if key == "mask":
                    key = "image_mask"

                try:
                    value = getattr(p, key)
                    if value in (None, []):
                        # no need to payloading
                        continue
                    if key == "init_images":
                        # init_images must be base64 encorded
                        payload[key] = convert(value)
                        continue
                    if key == "image_mask":
                        # payload need "mask" not "image_mask"
                        payload["mask"] = convert([value])[0]
                        continue
                    if key == "script_args":
                        # not support yet
                        continue
                    payload[key] = value

                except AttributeError:
                    # no attr in p with same name as key
                    continue

        # run request
        run_requests = []
        for i in range(WORKERS):
            request_input = {
                "input": {
                    "is_img2img": is_img2img,
                    "payload": payload,
                }
            }
            run_request = endpoint.run(request_input)
            run_requests.append(run_request)
            print(f"queued {run_request.job_id}")

        def watch(i, run_request, timer):
            DELAY = 1
            while True:
                status = run_request.status()
                timer.title = f"worker{i}"
                timer.status = status

                if (
                    shared.state.interrupted or
                    shared.state.stopping_generation
                ):
                    for run_request in run_requests:
                        run_request.cancel()

                if status == "COMPLETED":
                    break
                if status == "FAILED":
                    for run_request in run_requests:
                        run_request.cancel()
                    raise Exception("FAILED")
                if status == "CANCELLED":
                    raise Exception("CANCELLED")

                time.sleep(DELAY)

        def fetch(i, run_request, timer):
            job_status = run_request._fetch_job()
            timer.title = f"fetch{i} {run_request.job_id}"
            output: str = "".join(job_status.get("output"))
            data = json.loads(output)
            images_base64 = data.get("images", [])
            images_pil = b64_imgs_convert_to_pil(images_base64)

            infotexts = []
            for i, image_pil in enumerate(images_pil):
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
                    print(f"saved {i}@{run_request.job_id}")

            job_times = {
                key: job_status.get(key) for key
                in ["delayTime", "executionTime"]
            }
            delay_time, execution_time = job_times.values()
            infotexts = [
                infotext
                + f", delayTime: {delay_time / 1000:.2f}s, "
                + f"executionTime: {execution_time / 1000:.2f}s"
                for infotext in infotexts
            ]

            return images_pil, infotexts

        with Timer("watch workers") as timer:
            watch_threads = [
                Thread(
                    target=watch,
                    args=(i, run_request, timer)
                )
                for i, run_request
                in enumerate(run_requests)
            ]
            for watch_thread in watch_threads:
                watch_thread.start()
            for watch_thread in watch_threads:
                watch_thread.join()

        with Timer("fetch results") as timer:
            fetch_threads = [
                ReturnableThread(
                    target=fetch,
                    args=(i, run_request, timer)
                )
                for i, run_request
                in enumerate(run_requests)
            ]
            for fetch_thread in fetch_threads:
                fetch_thread.start()
            results = [
                fetch_thread.join()
                for fetch_thread in fetch_threads
            ]

        all_img = []
        all_txt = []
        for images_pil, infotexts in results:
            all_img += images_pil
            all_txt += infotexts

        return Processed(p, all_img, infotexts=all_txt)
