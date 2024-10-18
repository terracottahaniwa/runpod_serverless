import base64

from PIL import Image


def is_image(value):
    return isinstance(value, Image.Image)


def is_base64(value):
    try:
        base64.b64decode(value)
        return True
    except Exception:
        return False


def is_list_images(value):
    if type(value) != list:
        return False
    for item in value:
        if not is_image(item):
            return False
    return True


def is_list_base64(value):
    if type(value) != list:
        return False
    for item in value:
        if not is_base64(item):
            return False
    return True
