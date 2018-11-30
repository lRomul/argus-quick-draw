import json
import tqdm
import time
import random
import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config


def get_train_val_samples(val_key_id_path, blacklist_path=None):
    with open(val_key_id_path) as file:
        val_key_id_set = set(json.loads(file.read()))

    blacklist = set()
    if blacklist_path is not None:
        with open(blacklist_path) as file:
            blacklist = set(json.loads(file.read()))

    train_drawing_lst = []
    train_class_lst = []
    train_country_lst = []
    val_drawing_lst = []
    val_class_lst = []
    val_country_lst = []

    for cls in tqdm.tqdm(config.CLASSES):
        class_df = pd.read_csv(config.CLASS_TO_CSV_PATH[cls])
        class_df = class_df[~class_df.key_id.isin(blacklist)]
        val_key_ids = class_df.key_id.isin(val_key_id_set)

        train_class_df = class_df[~val_key_ids]
        train_drawings = train_class_df.drawing.values
        train_words = train_class_df.word.values
        train_countries = train_class_df.countrycode.values
        train_drawing_lst += train_drawings.tolist()
        train_class_lst += train_words.tolist()
        train_country_lst += train_countries.tolist()

        val_class_df = class_df[val_key_ids]
        val_drawings = val_class_df.drawing.values
        val_words = val_class_df.word.values
        val_countries = val_class_df.countrycode.values
        val_drawing_lst += val_drawings.tolist()
        val_class_lst += val_words.tolist()
        val_country_lst += val_countries.tolist()

    train_samples = train_drawing_lst, train_class_lst, train_country_lst
    val_samples = val_drawing_lst, val_class_lst, val_country_lst

    return train_samples, val_samples


class DrawDataset(Dataset):
    def __init__(self, samples,
                 draw_transform,
                 size=None,
                 image_transform=None):
        super().__init__()
        self.image_transform = image_transform
        self.draw_transform = draw_transform
        self.size = size

        self.drawing_lst, self.class_lst, self.country_lst = samples

    def __len__(self):
        if self.size is None:
            return len(self.drawing_lst)
        else:
            return self.size

    def __getitem__(self, idx):
        if self.size is not None:
            seed = int(time.time() * 1000.0) + idx
            random.seed(seed)
            idx = random.randint(0, len(self.drawing_lst) - 1)

        drawing = eval(self.drawing_lst[idx])
        cls = self.class_lst[idx]
        country = str(self.country_lst[idx])

        image = self.draw_transform(drawing)
        if self.image_transform is not None:
            image = self.image_transform(image)

        cls_idx = torch.tensor(config.CLASS_TO_IDX[cls])
        country_idx = torch.tensor(config.COUNTRY_TO_IDX[country])
        return (image, country_idx), cls_idx
