import gc
import torch
import collections
from argus import Model
from argus.utils import detach_tensors
from cnn_finetune import make_model

from src.nn_modules import CountryEmbModel
from src.metrics import MAPatK


class CnnFinetune(Model):
    nn_module = make_model


class DrawMetaModel(Model):
    nn_module = {
        'cnn_finetune': make_model,
        'CountryEmbModel': CountryEmbModel
    }


class IterSizeMetaModel(Model):
    nn_module = {
        'CountryEmbModel': CountryEmbModel
    }

    def train_step(self, batch) -> dict:
        if not self.nn_module.training:
            self.nn_module.train()
        self.optimizer.zero_grad()

        inputs, targets = batch
        imgs, countries = inputs

        imgs = torch.chunk(imgs, self.params['iter_size'], dim=0)
        countries = torch.chunk(countries, self.params['iter_size'], dim=0)
        targets = torch.chunk(targets, self.params['iter_size'], dim=0)

        for img, country, target in zip(imgs, countries, targets):
            input, target = self.prepare_batch(((img, country), target), self.device)
            prediction = self.nn_module(input)
            loss = self.loss(prediction, target)
            loss.backward()

        self.optimizer.step()

        prediction = detach_tensors(prediction)
        target = detach_tensors(target)
        del imgs, countries, targets
        gc.collect()

        return {
            'prediction': self.prediction_transform(prediction),
            'target': target,
            'loss': loss.item()
        }
