import torch
import numpy as np
import sklearn.metrics

def acc_t(truth,pred):
    return sklearn.metrics.accuracy_score(truth.cpu().numpy(),pred.cpu().numpy())

def performance(acc):
    1

def f1_score(truth, pred, average):
    return sklearn.metrics.f1_score(truth.cpu().numpy(), pred.cpu().numpy(), average=average)