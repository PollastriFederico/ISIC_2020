import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import torch

# if torch.__version__ != '1.1.0':
#     raise RuntimeError('PyTorch version must be 1.1.0')
from torch.nn.functional import softmax


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.conf_matrix = np.zeros((num_classes, num_classes), int)

    def update_matrix(self, out, target):
        # I'm sure there is a better way to do this
        for j in range(len(target)):
            self.conf_matrix[out[j].item(), target[j].item()] += 1

    def get_metrics(self):
        samples_for_class = np.sum(self.conf_matrix, 0)
        diag = np.diagonal(self.conf_matrix)

        try:
            acc = np.sum(diag) / np.sum(samples_for_class)
            w_acc = np.divide(diag, samples_for_class)
            w_acc = np.mean(w_acc)
        except:
            acc = 0
            w_acc = 0

        return acc, w_acc


# compute one_hot format lables
def categorical_to_one_hot(t, max_val):
    one_hot = torch.zeros(t.size(0), max_val)
    one_hot.scatter_(1, t.view(-1, 1), 1)
    return one_hot


'''
Differential Entropy
'''


def entropy_categorical(p):
    '''
    Differential entropy of categorical distribution
        input: matrix of shape (batch,probs) where probs is the probability for each of the outcomes
        output: matrix of shape (batch,) with the entropy per each value
    '''
    return (-1 * p * torch.log(p + 1e-34)).sum(-1)
