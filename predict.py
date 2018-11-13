import torch
import numpy as np
import pandas as pd
import tqdm
from os.path import join

from argus import load_model

from src.transforms import ImageTransform, DrawTransform
from src.argus_models import DrawMetaModel
from src.utils import make_dir
from src import config


DRAW_SIZE = 128
DRAW_PAD = 4
DRAW_LINE_WIDTH = 2
TIME_COLOR = True
SCALE_SIZE = 128
PRED_BATCH_SIZE = 1024
MODEL_PATH = '/workdir/data/experiments/rainbow_country_se_resnext50_001/model-109-0.885694.pth'
PREDICT_DIR = '/workdir/data/predictions/rainbow_country_se_resnext50_001'


class Predictor:
    def __init__(self, model_path, draw_transform, image_transform):
        self.model = load_model(model_path)
        self.model.nn_module.eval()

        self.draw_transform = draw_transform
        self.image_transform = image_transform

    def __call__(self, samples):
        tensors = []
        country_tensors = []
        for drawing, country in samples:
            image = self.draw_transform(drawing)
            tensor = self.image_transform(image)
            if country == 'OTHER':
                country = 'nan'
            country = torch.tensor(config.COUNTRY_TO_IDX[country])
            country_tensors.append(country)
            tensors.append(tensor)

        tensor = torch.stack(tensors, dim=0)
        tensor = tensor.to(self.model.device)

        country_tensor = torch.stack(country_tensors, dim=0)
        country_tensor = country_tensor.to(self.model.device)

        with torch.no_grad():
            probs = self.model.predict((tensor, country_tensor))
            return probs


if __name__ == "__main__":
    make_dir(PREDICT_DIR)
    draw_transform = DrawTransform(DRAW_SIZE, DRAW_PAD, DRAW_LINE_WIDTH, TIME_COLOR)
    image_trns = ImageTransform(False, SCALE_SIZE)

    test_df = pd.read_csv(config.TEST_SIMPLIFIED_PATH)
    sample_subm = pd.read_csv(config.SAMPLE_SUBMISSION)
    predictor = Predictor(MODEL_PATH, draw_transform, image_trns)

    drawings = []
    key_ids = []
    pred_words = []
    pred_key_ids = []
    probs_lst = []
    for i, row in tqdm.tqdm(test_df.iterrows()):
        drawing = eval(row.drawing)

        drawings.append((drawing, str(row.countrycode)))
        key_ids.append(row.key_id)
        if len(drawings) == PRED_BATCH_SIZE:
            probs = predictor(drawings).cpu().numpy()
            probs_lst.append(probs)

            preds_idx = probs.argsort(axis=1)
            preds_idx = np.fliplr(preds_idx)[:, :3]
            for pred_idx, key_id in zip(preds_idx, key_ids):
                words = [config.IDX_TO_CLASS[i].replace(' ', '_') for i in pred_idx]
                pred_words.append(" ".join(words))
                pred_key_ids.append(key_id)

            drawings = []
            key_ids = []

    probs = predictor(drawings).cpu().numpy()
    preds_idx = probs.argsort(axis=1)
    preds_idx = np.fliplr(preds_idx)[:, :3]
    for pred_idx, key_id in zip(preds_idx, key_ids):
        words = [config.IDX_TO_CLASS[i].replace(' ', '_') for i in pred_idx]
        pred_words.append(" ".join(words))
        pred_key_ids.append(key_id)

    drawings = []
    key_ids = []

    probs_array = np.stack(probs_lst, axis=0)
    np.save(join(PREDICT_DIR, 'probs.npy'), probs_array)

    submission = pd.DataFrame({'key_id': pred_key_ids, 'word': pred_words})
    submission.to_csv(join(PREDICT_DIR, 'submission.csv'), index=False)
