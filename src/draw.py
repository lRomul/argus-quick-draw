import cv2
import numpy as np
import matplotlib.pyplot as plt

from src import config


COLOR_MAP = plt.get_cmap('brg')


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


def draw_cv2(drawing, size=128, lw=2, shift=4, time_color=True):
    img = np.zeros((size, size, 3), np.uint8)
    for t, stroke in enumerate(drawing):
        for i in range(len(stroke[0]) - 1):
            if time_color:
                color = t / max(1, len(drawing) - 1)
                color = tuple([int(c * 255) for c in COLOR_MAP(color)[:3]])
            else:
                color = (255, 255, 255)
            cv2.line(img, (stroke[0][i] + shift, stroke[1][i] + shift),
                     (stroke[0][i + 1] + shift, stroke[1][i + 1] + shift), color, lw)
    return img
