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
        size = np.random.randint(self.size[0], self.size[1])
        img = random_crop(img, (size, size))
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
        self.size = (size, size)
        self.interpolation = interpolation

    def __call__(self, img):
        if img.shape[:2] != self.size:
            img = cv2.resize(img,
                             dsize=self.size,
                             interpolation=self.interpolation)
        return img


class RandomBorderScale:
    def __init__(self, max_scale=0.7):
        self.max_scale = max_scale

    def __call__(self, img):
        scale = random.uniform(self.max_scale, 1.0)
        size = img.shape[:2][::-1]
        scale_img_size = (np.array(size) * scale).astype(int)
        scale_img_size = tuple(scale_img_size.tolist())
        scale_img = cv2.resize(img, scale_img_size)
        border_image = np.zeros_like(img)

        x_shift, y_shift = (np.array(size) - scale_img_size) // 2
        border_image[y_shift:y_shift + scale_img.shape[0],
                     x_shift:x_shift + scale_img.shape[1]] = scale_img

        return border_image


class Rotate:
    def __init__(self, limit=15):
        self.limit = limit

    def __call__(self, img):
        angle = random.uniform(-self.limit, self.limit)

        height, width = img.shape[0:2]
        mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        rotate_img = cv2.warpAffine(img, mat, (height, width),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
        return rotate_img


class ImageToTensor:
    def __call__(self, image):
        image = np.moveaxis(image, -1, 0)
        image = image.astype(np.float32) / 255.0
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
    def __init__(self, train, scale_size):
        self.train = train

        if train:
            self.transform = Compose([
                # UseWithProb(HorizontalFlip(), 0.3),
                # UseWithProb(RandomCrop(random_crop_size), 0.2),
                # UseWithProb(RandomBorderScale(max_border_scale), 0.2),
                # UseWithProb(Rotate(rotate_angle), 0.5),
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
