import io
from base64 import (
    b64encode,
    b64decode,
)

from PIL import Image

from scripts.check import (
    is_image,
    is_base64,
    is_list_images,
    is_list_base64,
)


def to_base64(image):
    assert is_image(image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    content = buffer.getvalue()
    return b64encode(content).decode("utf-8")


def to_image(base64_):
    assert is_base64(base64_)
    content = b64decode(base64_)
    buffer = io.BytesIO(content)
    return Image.open(buffer)


def list_to_base64(images):
    assert is_list_images(images)
    return [
        to_base64(image)
        for image in images
    ]


def list_to_image(base64s):
    assert is_list_base64(base64s)
    return [
        to_image(base64_)
        for base64_ in base64s
    ]
