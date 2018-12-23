from torch.utils.data import DataLoader

from argus.callbacks import MonitorCheckpoint, EarlyStopping
from argus.callbacks import LoggingToFile, ReduceLROnPlateau

from src.datasets import DrawDataset, get_train_val_samples
from src.transforms import ImageTransform, DrawTransform
from src.argus_models import CnnFinetune, DrawMetaModel, IterSizeMetaModel
from src.metrics import MAPatK
from src import config


DRAW_SIZE = 256
SCALE_SIZE = 128
DRAW_PAD = 3
DRAW_LINE_WIDTH = 3
TIME_COLOR = True
ITER_SIZE = 5
BATCH_SIZE = 208 * ITER_SIZE
TRAIN_EPOCH_SIZE = 1000000
VAL_KEY_ID_PATH = '/workdir/data/val_key_ids_001.json'
EXPERIMENT_NAME = 'iter_size_se_resnext50_001'

PARAMS = {
    'nn_module': ('CountryEmbModel', {
        'cnn_finetune': {
            'model_name': 'se_resnext50_32x4d',
            'num_classes': len(config.CLASSES),
            'pretrained': True,
            'dropout_p': 0.2
        },
        'num_country': len(config.COUNTRIES),
        'country_emb_dim': 10
    }),
    'iter_size': ITER_SIZE,
    'optimizer': ('Adam', {'lr': 0.001}),
    'loss': 'CrossEntropyLoss',
    'device': 'cuda'
}


if __name__ == "__main__":
    train_samples, val_samples = get_train_val_samples(VAL_KEY_ID_PATH)

    draw_transform = DrawTransform(DRAW_SIZE, DRAW_PAD, DRAW_LINE_WIDTH, TIME_COLOR)
    train_trns = ImageTransform(True, SCALE_SIZE)
    train_dataset = DrawDataset(train_samples, draw_transform,
                                size=TRAIN_EPOCH_SIZE, image_transform=train_trns)
    val_trns = ImageTransform(False, SCALE_SIZE)
    val_dataset = DrawDataset(val_samples, draw_transform, image_transform=val_trns)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            num_workers=8, shuffle=False)

    model = IterSizeMetaModel(PARAMS)

    callbacks = [
        MonitorCheckpoint(f'/workdir/data/experiments/{EXPERIMENT_NAME}',
                          monitor='val_map_at_k', max_saves=10),
        EarlyStopping(monitor='val_map_at_k', patience=50),
        ReduceLROnPlateau(monitor='val_map_at_k', factor=0.64,
                          patience=1, min_lr=0.000001),
        LoggingToFile(f'/workdir/data/experiments/{EXPERIMENT_NAME}/log.txt')
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=1000,
              callbacks=callbacks,
              metrics=['accuracy', MAPatK(k=3)])
