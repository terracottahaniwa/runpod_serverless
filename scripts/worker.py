import json
import time

import modules.shared as shared
import modules.images
from  modules.infotext_utils import (
    parse_generation_parameters
)

from scripts.convert import list_to_image


def cancel_all(run_requests):
    for run_request in run_requests:
        run_request.cancel()


def watch_status(i, run_requests, timer):
    DELAY = 1
    run_request= run_requests[i]
    while True:
        if any((shared.state.interrupted, 
                shared.state.stopping_generation)):
            cancel_all(run_requests)
    
        status = run_request.status()
        timer.title = f"worker{i}"
        timer.status = status

        if status == "FAILED":
            cancel_all(run_requests)
            return "FAILED"
        if status == "CANCELLED":
            return "CANCELLED"
        if status == "COMPLETED":
            return "COMPLETED"

        time.sleep(DELAY)


def get_job_times(job_status):
    job_times = {
        key: job_status.get(key) for key
        in ["delayTime", "executionTime"]
    }
    delay_time, execution_time = job_times.values()
    return delay_time, execution_time


def fetch_result(i, p, run_requests, timer):
    run_request = run_requests[i]
    job_status = run_request._fetch_job()

    timer.title = f"fetch{i}"
    timer.status = f"{run_request.job_id}"

    delay_time, execution_time = get_job_times(job_status)
    job_times_info = (
        f"delayTime: {delay_time / 1000:.2f}s, "
        f"executionTime: {execution_time / 1000:.2f}s"
    )
    print(f"worker{i}: done. {job_times_info}")

    output = "".join(job_status.get("output"))
    data = json.loads(output)
    base64_ = data.get("images", [])
    images = list_to_image(base64_)
    infotexts = []

    for j, image in enumerate(images):
        infotext = image.text.get("parameters")
        infotexts.append(infotext)

        info = parse_generation_parameters(infotext)
        is_save_sample = p.save_samples()
        if is_save_sample:
            modules.images.save_image(
                image=image,
                path=p.outpath_samples,
                basename="runpod",
                seed=info.get("Seed"),
                prompt=info.get("Prompt"),
                extension=shared.opts.samples_format,
                info=infotext,
                p=p
            )
            print(
                f"saved: image{j} @ worker{i} "
                f"<{run_request.job_id}>"
            )

    infotexts = [
        infotext + ", " + job_times_info
        for infotext in infotexts
    ]

    return images, infotexts
