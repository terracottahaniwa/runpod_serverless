import os
from datetime import datetime

import runpod
import gradio as gr

import modules.shared as shared
import modules.scripts as scripts
from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
    Processed,
)

from scripts.payload import (
    load_template,
    create_payload,
    create_request_input,
)
from scripts.custom_class import (
    CounterTimer,
    ReturnableThread,
)
from scripts.worker import (
    watch_status,
    fetch_result,
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
            type="password",
        )
        RUNPOD_ENDPOINT_ID = gr.Textbox(
            max_lines=1,
            label="RUNPOD_ENDPOINT_ID",
            value=os.getenv("RUNPOD_ENDPOINT_ID"),
        )
        WORKERS = gr.Slider(
            minimum=1,
            maximum=10,
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

        template = load_template(BASEDIR, is_img2img)
        payload = create_payload(p, template)
        request_input = create_request_input(is_img2img, payload)

        now = datetime.now().strftime(r"%y/%m/%d %H:%M:%S")
        print(f"\nRunpod serverless run: {now}")

        run_requests = []
        for i in range(WORKERS):
            run_request = endpoint.run(request_input)
            run_requests.append(run_request)
            print(f"worker{i}: {run_request.job_id}")

        def update_progress_bar(timer):
            shared.state.textinfo = (
                f"{timer.title}: "
                f"{timer.counter}s "
                f"{timer.status}"
            )

        with CounterTimer() as timer:
            timer.title = "watch status"
            timer.status = ""
            timer.hook = update_progress_bar
            watch_threads = [
                ReturnableThread(
                    target=watch_status,
                    args=(i, run_requests, timer)
                )
                for i in range(len(run_requests))
            ]
            for watch_thread in watch_threads:
                watch_thread.start()
            results = [
                watch_thread.join()
                for watch_thread in watch_threads
            ]
        
        assert all(
            [
                result == "COMPLETED"
                for result in results
            ]
        ), results

        with CounterTimer() as timer:
            timer.title  = "fetch results"
            timer.status = ""
            timer.hook = update_progress_bar
            fetch_threads = [
                ReturnableThread(
                    target=fetch_result,
                    args=(i, p, run_requests, timer)
                )
                for i in range(len(run_requests))
            ]
            for fetch_thread in fetch_threads:
                fetch_thread.start()
            results = [
                fetch_thread.join()
                for fetch_thread in fetch_threads
            ]

        all_images = []
        all_infotexts = []
        for images, infotexts in results:
            all_images += images
            all_infotexts += infotexts

        now = datetime.now().strftime(r"%y/%m/%d %H:%M:%S")
        print(f"Runpod serverless finish: {now}")
        return Processed(p, all_images, infotexts=all_infotexts)
