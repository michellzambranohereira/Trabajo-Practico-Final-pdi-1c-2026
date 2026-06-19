import cv2
import numpy as np
from PIL import Image

def equalization(image):
    img = np.array(image.convert("L"))
    equalized = cv2.equalizeHist(img)
    return Image.fromarray(equalized)

def gaussian_blur(image):
    img = np.array(image)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    return Image.fromarray(blurred)

def apply_clahe(image):
    img = np.array(image.convert("L"))

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    result = clahe.apply(img)

    return Image.fromarray(result)