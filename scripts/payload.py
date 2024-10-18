import json
import os

from modules.processing import (
    StableDiffusionProcessing,
)

from scripts.convert import (
    to_base64,
    list_to_base64,
)


def load_template(basedir, is_img2img):
    subdir = "template"
    if is_img2img:
        filename = "img2img.json"
    else:
        filename = "txt2img.json"
    path = os.path.join(basedir, subdir, filename)
    with open(path) as f:
        template = json.load(f)
    return template


def create_payload(
        p: StableDiffusionProcessing, 
        template
    ):
    payload = {}
    for key in template.keys():
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
                payload[key] = list_to_base64(value)
                continue
            if key == "image_mask":
                # payload need "mask" not "image_mask"
                payload["mask"] = to_base64(value)
                continue
            if key == "script_args":
                # not support yet
                continue
            payload[key] = value

        except AttributeError:
            # no attr in p with same name as key
            continue
    return payload


def create_request_input(is_img2img, payload):
    request_input = {
        "input": {
            "is_img2img": is_img2img,
            "payload": payload,
        }
    }
    return request_input
