from argus import Model
from cnn_finetune import make_model

from src.nn_modules import CountryEmbModel


class CnnFinetune(Model):
    nn_module = make_model


class DrawMetaModel(Model):
    nn_module = {
        'cnn_finetune': make_model,
        'CountryEmbModel': CountryEmbModel
    }
