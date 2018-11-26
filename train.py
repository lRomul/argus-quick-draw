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
from src.argus_models import CnnFinetune, DrawMetaModel
from src.metrics import MAPatK
from src import config


DRAW_SIZE = 256
SCALE_SIZE = 256
DRAW_PAD = 6
DRAW_LINE_WIDTH = 3
TIME_COLOR = True
TRAIN_BATCH_SIZE = 102
VAL_BATCH_SIZE = 102
TRAIN_EPOCH_SIZE = 1000000
LR = 2e-5
N_WORKERS = 8
EXP_DIR = '/workdir/data/experiments/'
VAL_KEY_ID_PATH = '/workdir/data/val_key_ids_001.json'
PRETRAIN_PATH = f'{EXP_DIR}/rb_ctry_se_resnext50_002/model-013-0.887236.pth'
EXP_NAME = 'rb_ctry_se_resnext50_002a'


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
    'optimizer': ('Adam', {'lr': LR}),
    'loss': 'CrossEntropyLoss',
    'device': ['cuda:0', 'cuda:1']
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
                              num_workers=N_WORKERS, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE,
                            num_workers=N_WORKERS, shuffle=False)

    model = DrawMetaModel(params)

    if PRETRAIN_PATH is not None:
        model = load_model(PRETRAIN_PATH)
        model.set_lr(LR)
        model._build_device(params)

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_map_at_k', max_saves=3),
        EarlyStopping(monitor='val_map_at_k', patience=45),
        ReduceLROnPlateau(monitor='val_map_at_k', factor=0.75, patience=1,
                          min_lr=0.000001),
        LoggingToFile(f'{save_dir}/log.txt')
    ]
    
    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=1000,
              callbacks=callbacks,
              metrics=['accuracy', MAPatK(k=3)])

