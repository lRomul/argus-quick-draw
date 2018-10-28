import cv2
import numpy as np
from typing import List

from src import config


def scale_drawing(drawing, size=112):
    x_max = 0
    y_max = 0
    for x_lst, y_lst in drawing:
        x_max = max(x_max, max(x_lst))
        y_max = max(y_max, max(y_lst))

    x_shift = (config.BASE_SIZE_SIMPLIFIED - x_max) // 2
    y_shift = (config.BASE_SIZE_SIMPLIFIED - y_max) // 2

    scaled_drawing = []
    for x_lst, y_lst in drawing:
        x_scaled_lst = [round(((x + x_shift) / config.BASE_SIZE_SIMPLIFIED) * size) for x in x_lst]
        y_scaled_lst = [round(((y + y_shift) / config.BASE_SIZE_SIMPLIFIED) * size) for y in y_lst]
        scaled_drawing.append([x_scaled_lst, y_scaled_lst])

    return scaled_drawing


def draw_cv2(drawing, size=128, lw=3, shift=8, time_color=True):
    img = np.zeros((size, size), np.uint8)
    for t, stroke in enumerate(drawing):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i] + shift, stroke[1][i] + shift),
                         (stroke[0][i + 1] + shift, stroke[1][i + 1] + shift), color, lw)
    return img
