import os
import cv2
import numpy as np

from torch.utils.data import DataLoader

import argus
from argus import Model
from argus import load_model
from argus.callbacks import MonitorCheckpoint, EarlyStopping
from argus.callbacks import LoggingToFile, ReduceLROnPlateau

from src.utils import make_dir
from src.datasets import DrawDataset, get_train_val_samples
from src.transforms import ImageTransform, DrawTransform
from src.argus_models import CnnFinetune, DrawMetaModel, IterSizeMetaModel
from src.metrics import MAPatK
from src import config


DRAW_SIZE = 128
SCALE_SIZE = 128
DRAW_PAD = 4
DRAW_LINE_WIDTH = 2
TIME_COLOR = True
ITER_SIZE = 3
TRAIN_BATCH_SIZE = 448 * ITER_SIZE
VAL_BATCH_SIZE = 448 * ITER_SIZE
TRAIN_EPOCH_SIZE = 1000000
LR = 2e-5
N_WORKERS = 8
EXP_DIR = '/workdir/data/experiments/'
VAL_KEY_ID_PATH = '/workdir/data/val_key_ids_001.json'
PRETRAIN_PATH = f'{EXP_DIR}/rainbow_country_se_resnext50_001/model-109-0.885694.pth'
EXP_NAME = 'iter_rb_ctry_se_resnext50_004'


params = {
    'nn_module': ('CountryEmbModel' ,{
        'cnn_finetune': {
            'model_name': 'se_resnext50_32x4d',
            'num_classes': len(config.CLASSES),
            'pretrained': True,
            'dropout_p': 0.2,
        },
        'num_country': len(config.COUNTRIES),
        'country_emb_dim': 10
    }),
    'iter_size': ITER_SIZE,
    'optimizer': ('Adam', {'lr': LR}),
    'loss': 'CrossEntropyLoss',
    'device': 'cuda:0'
}


if __name__ == "__main__":
    save_dir = os.path.join(EXP_DIR, EXP_NAME)
    make_dir(save_dir)
    with open(os.path.join(save_dir, 'source.py'), 'w') as outfile:
        outfile.write(open(__file__).read())
    train_samples, val_samples = get_train_val_samples(VAL_KEY_ID_PATH)
    draw_transform = DrawTransform(DRAW_SIZE, DRAW_PAD, DRAW_LINE_WIDTH,
                                   TIME_COLOR)
    train_trns = ImageTransform(True, SCALE_SIZE)
    train_dataset = DrawDataset(train_samples, draw_transform,
                                size=TRAIN_EPOCH_SIZE,
                                image_transform=train_trns)
    val_trns = ImageTransform(False, SCALE_SIZE)
    val_dataset = DrawDataset(val_samples, draw_transform,
                              image_transform=val_trns)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE,
                              num_workers=N_WORKERS, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE,
                            num_workers=N_WORKERS, shuffle=False, pin_memory=True)

    model = IterSizeMetaModel(params)

    if PRETRAIN_PATH is not None:
        pretrain_model = load_model(PRETRAIN_PATH)
        model.nn_module.load_state_dict(
            pretrain_model.nn_module.state_dict())
        model.set_lr(LR)
        params['device'] = ['cuda:0', 'cuda:1']
        model._build_device(params)
        del pretrain_model

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_map_at_k', max_saves=3),
        EarlyStopping(monitor='val_map_at_k', patience=45),
        ReduceLROnPlateau(monitor='val_map_at_k', factor=0.75, patience=1,
                          min_lr=1e-7),
        LoggingToFile(f'{save_dir}/log.txt')
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=1000,
              callbacks=callbacks,
              metrics=['accuracy', MAPatK(k=3)])
