import cv2
import numpy as np
from PIL import Image


# Ecualización del histograma
def equalization(image):

    img = np.array(image.convert("L"))

    equalized = cv2.equalizeHist(img)

    return Image.fromarray(equalized)


# Suavizado Gaussiano
def gaussian_blur(image):

    img = np.array(image)

    blurred = cv2.GaussianBlur(
        img,
        (5, 5),
        0
    )

    return Image.fromarray(blurred)


# CLAHE
def apply_clahe(image):

    img = np.array(image.convert("L"))

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    result = clahe.apply(img)

    return Image.fromarray(result)


# Brillo y contraste
def adjust_brightness_contrast(
        image,
        alpha=1.4,
        beta=20):

    img = np.array(image)

    result = cv2.convertScaleAbs(
        img,
        alpha=alpha,   # contraste
        beta=beta      # brillo
    )

    return Image.fromarray(result)

## HSV - Canal H

def hsv_h(image):

    img = np.array(image)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]

    h_eq = cv2.equalizeHist(h)

    hsv_mejorado = hsv.copy()
    hsv_mejorado[:, :, 0] = h_eq

    rgb_final = cv2.cvtColor(
        hsv_mejorado,
        cv2.COLOR_HSV2RGB
    )

    return Image.fromarray(rgb_final)

# HSV - Canal S

def hsv_s(image):

    img = np.array(image)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    s_eq = cv2.equalizeHist(s)

    hsv_mejorado = hsv.copy()
    hsv_mejorado[:, :, 1] = s_eq

    rgb_final = cv2.cvtColor(
        hsv_mejorado,
        cv2.COLOR_HSV2RGB
    )

    return Image.fromarray(rgb_final)

# HSV - Canal V

def hsv_v(image):

    img = np.array(image)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    v = hsv[:, :, 2]

    v_eq = cv2.equalizeHist(v)

    hsv_mejorado = hsv.copy()
    hsv_mejorado[:, :, 2] = v_eq

    rgb_final = cv2.cvtColor(
        hsv_mejorado,
        cv2.COLOR_HSV2RGB
    )

    return Image.fromarray(rgb_final)