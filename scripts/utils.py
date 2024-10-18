import base64
import io
import time
from threading import Thread

from PIL import Image

import modules.shared as shared


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
    assert is_pil_images(images), images
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
    assert is_b64_images(images), images
    images_pil = []
    for image_base64 in images:
        image_bytes = base64.b64decode(image_base64)
        image_bytesio = io.BytesIO(image_bytes)
        image_pil = Image.open(image_bytesio)
        images_pil.append(image_pil)
    return images_pil


class Timer():
    def __init__(self, title="", status=""):
        self.title = title
        self.status = status
        self.is_running = False
        self.counter = 0
        self.thread = Thread(
            target=self.timer
        )

    def timer(self):
        DELAY = 1
        while self.is_running:
            shared.state.textinfo = (
                f"{self.title}: "
                f"{self.counter}s "
                f"{self.status}"
            )
            time.sleep(DELAY)
            self.counter += DELAY

    def __enter__(self):
        self.is_running = True
        self.thread.start()
        return self

    def __exit__(self, ex_type, ex_value, trace):
        self.is_running = False
        self.thread.join()


class ReturnableThread(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._return = None

    def run(self):
        self._return = self._target(
            *self._args,
            **self._kwargs
        )

    def join(self):
        super().join()
        return self._return