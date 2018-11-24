import os
import cv2
import numpy as np

from torch.utils.data import DataLoader

import argus
from argus import Model
from argus import load_model
from argus.callbacks import MonitorCheckpoint, EarlyStopping
from argus.callbacks import LoggingToFile, ReduceLROnPlateau

from src.datasets import DrawDataset, get_train_val_samples
from src.transforms import ImageTransform, DrawTransform
from src.argus_models import CnnFinetune, DrawMetaModel
from src.metrics import MAPatK
from src import config


image_size = 256
scale_size = 256
image_pad = 6
image_line_width = 3
time_color = True
train_batch_size = 48
val_batch_size = 48
train_epoch_size = 1000000
val_key_id_path = '/workdir/data/val_key_ids_001.json'
pretrain_path = '/workdir/data/experiments/rainbow_country_se_resnext50_001_after_001/model-063-0.887085.pth'
experiment_name = 'rb_ctry_se_resnext50_test'


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
    'optimizer': ('Adam', {'lr': 0.001}),
    'loss': 'CrossEntropyLoss',
    'device': 'cuda'
}

train_samples, val_samples = get_train_val_samples(val_key_id_path)

draw_transform = DrawTransform(image_size, image_pad, image_line_width, time_color)
train_trns = ImageTransform(True, scale_size)
train_dataset = DrawDataset(train_samples, draw_transform,
                            size=train_epoch_size, image_transform=train_trns)
val_trns = ImageTransform(False, scale_size)
val_dataset = DrawDataset(val_samples, draw_transform, image_transform=val_trns)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=8, shuffle=False)

model = DrawMetaModel(params)

if pretrain_path is not None:
    model = load_model(pretrain_path)

callbacks = [
    MonitorCheckpoint(f'/workdir/data/experiments/{experiment_name}', monitor='val_map_at_k', max_saves=3),
    EarlyStopping(monitor='val_map_at_k', patience=45),
    ReduceLROnPlateau(monitor='val_map_at_k', factor=0.75, patience=1, min_lr=0.000001),
    LoggingToFile(f'/workdir/data/experiments/{experiment_name}/log.txt')
]

model.fit(train_loader,
          val_loader=val_loader,
          max_epochs=1000,
          callbacks=callbacks,
          metrics=['accuracy', MAPatK(k=3)])

