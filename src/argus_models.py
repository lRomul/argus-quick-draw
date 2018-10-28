from argus import Model
from cnn_finetune import make_model


class CnnFinetune(Model):
    nn_module = make_model
