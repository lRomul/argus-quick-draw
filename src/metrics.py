import torch
import numpy as np

from argus.metrics import Metric


def apk(actual, predicted, k=3):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


class MAPatK(Metric):
    name = 'map_at_k'
    better = 'max'

    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.scores = []

    def reset(self):
        self.scores = []

    def update(self, step_output: dict):
        preds = step_output['prediction'].cpu().numpy()
        trgs = step_output['target'].cpu().numpy()

        preds_idx = preds.argsort(axis=1)
        preds_idx = np.fliplr(preds_idx)[:, :self.k]

        self.scores += [apk([a], p, self.k) for a, p in zip(trgs, preds_idx)]

    def compute(self):
        return np.mean(self.scores)
