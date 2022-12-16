import io
import cv2
import base64
from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt


def numpy_to_b64(img):
    """
    Convert image from numpy array to base64 string
    :param img: image as numpy array
    :return: image as base64 encoding
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.imencode('.jpg', img)
    b64_string = base64.b64encode(img[1]).decode('utf-8')
    return b64_string


def b64_to_numpy(img, filename=None):
    """
    Convert base64 string to numpy array image
    :param img: image as base64 encoding
    :param filename: filename to save the image
    :return: image as numpy array
    """
    img = np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(img)))))

    #im = ImageOps.exif_transpose(im)

    # if png, remove transparency channel
    if len(img.shape) > 2 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if filename:
        plt.imsave(filename, img)

    return img
