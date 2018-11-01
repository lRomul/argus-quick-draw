import torch
import random
import numpy as np

import cv2
cv2.setNumThreads(0)

from src.draw import scale_drawing, draw_cv2


def img_size(image: np.ndarray):
    return image.shape[1], image.shape[0]


def img_crop(img, box):
    return img[box[1]:box[3], box[0]:box[2]]


def random_crop(img, size):
    tw = size[0]
    th = size[1]
    w, h = img_size(img)
    if ((w - tw) > 0) and ((h - th) > 0):
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
    else:
        x1 = 0
        y1 = 0
    img_return = img_crop(img, (x1, y1, x1 + tw, y1 + th))
    return img_return


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for trns in self.transforms:
            image = trns(image)
        return image


class OneOf:
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            transform = random.choice(self.transforms)
            image = transform(image)
        return image


class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = self.transform(image)
        return image


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = random_crop(img, self.size)
        return img


class Flip:
    def __init__(self, flip_code):
        assert flip_code in [0, 1]
        self.flip_code = flip_code

    def __call__(self, img):
        img = cv2.flip(img, self.flip_code)
        return img


class HorizontalFlip(Flip):
    def __init__(self):
        super().__init__(1)


class Scale:
    def __init__(self, size, interpolation=cv2.INTER_AREA):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        img = cv2.resize(img,
                         dsize=(self.size, self.size),
                         interpolation=self.interpolation)
        return img


class ImageToTensor:
    def __call__(self, image):
        image = image.astype(np.float32) / 255.0
        image = np.stack([image]*3, axis=0)
        return torch.from_numpy(image)


class DrawTransform:
    def __init__(self, size=128, pad=8, line_width=3, time_color=True):
        self.size = size
        self.line_width = line_width
        self.time_color = time_color
        self.pad = pad
        self.scale_size = self.size - 2*self.pad

    def __call__(self, drawing):
        scaled_drawing = scale_drawing(drawing, self.scale_size)
        image = draw_cv2(scaled_drawing, size=self.size, lw=self.line_width,
                         shift=self.pad, time_color=self.time_color)
        return image


class ImageTransform:
    def __init__(self, train, scale_size=64):
        self.train = train

        if train:
            self.transform = Compose([
                UseWithProb(HorizontalFlip(), 0.4),
                Scale(scale_size),
                ImageToTensor()
            ])
        else:
            self.transform = Compose([
                Scale(scale_size),
                ImageToTensor()
            ])

    def __call__(self, image):
        image = self.transform(image)
        return image
