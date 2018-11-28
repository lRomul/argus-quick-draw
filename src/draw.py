import cv2
import numpy as np
import matplotlib.pyplot as plt

from src import config


COLOR_MAP = plt.get_cmap('rainbow')


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


def scale_time_drawing(drawing, size=112):
    x_max = 0
    y_max = 0
    for x_lst, y_lst, t in drawing:
        x_max = max(x_max, max(x_lst))
        y_max = max(y_max, max(y_lst))

    x_shift = (config.BASE_SIZE_SIMPLIFIED - x_max) // 2
    y_shift = (config.BASE_SIZE_SIMPLIFIED - y_max) // 2

    scaled_drawing = []
    for x_lst, y_lst, t in drawing:
        x_scaled_lst = [round(((x + x_shift) / config.BASE_SIZE_SIMPLIFIED) * size) for x in x_lst]
        y_scaled_lst = [round(((y + y_shift) / config.BASE_SIZE_SIMPLIFIED) * size) for y in y_lst]
        scaled_drawing.append([x_scaled_lst, y_scaled_lst, t])
    return scaled_drawing

t_col_scale = 0.1

def draw_time_cv2(drawing, size=128, lw=2, shift=4):
    img = np.zeros((size, size, 3), np.uint8)
    n_str = max(1, len(drawing))
    for t, stroke in enumerate(drawing):
        n_pt = len(stroke[0]) - 1
        color_stroke_s = t / n_str
        color_stroke_f = (t + 0.5) / n_str
        color_time_s = t_col_scale + (stroke[2][0] * (1 - t_col_scale))
        color_time_f = t_col_scale + (stroke[2][1] * (1 - t_col_scale))
        d_col = (color_stroke_f - color_stroke_s) / n_pt
        d_t = (color_time_f - color_time_s) / n_pt
        for i in range(n_pt):
            color_stroke = color_stroke_s + i * d_col
            color_stroke_inv = 1 - color_stroke
            color_time = color_time_s + i * d_t
            color = (color_stroke, color_stroke_inv, color_time)
            color = tuple([int(c * 255) for c in color])
            cv2.line(img, (stroke[0][i] + shift, stroke[1][i] + shift),
                     (stroke[0][i + 1] + shift, stroke[1][i + 1] + shift), color, lw)
    return img
