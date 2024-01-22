#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer script. Example run command: train.py save_to_folder configs/cnn.gin.
"""
import os
import gin
from gin.config import _CONFIG
import torch
import pickle
import logging
from functools import partial
logger = logging.getLogger(__name__)

from models.Greedy import dataset
from models.Greedy import callbacks as avail_callbacks
from models.Greedy.model import MMTM_MVCNN
from models.Greedy.training_loop import training_loop
from models.Greedy.utils import gin_wrap
import time

def blend_loss(y_hat, y):
    loss_func = torch.nn.CrossEntropyLoss()
    losses = []
    for y_pred in y_hat:
        losses.append(loss_func(y_pred, y))

    return sum(losses)


def acc(y_pred, y_true):
    if isinstance(y_pred, list):
        y_pred = torch.mean(torch.stack([out.data for out in y_pred], 0), 0)
    _, y_pred = y_pred.max(1)
    if len(y_true)==2:
        acc_pred = (y_pred == y_true[0]).float().mean()
    else:
        acc_pred = (y_pred == y_true).float().mean()
    return acc_pred * 100


@gin.configurable
def train(save_path, wd, lr, momentum, batch_size, callbacks=[]):
    model = MMTM_MVCNN()
    train_loader, valid, test = dataset.get_ks_data(batch_size, seed=0)

    optimizer = torch.optim.SGD(model.parameters(),
        lr=lr,
        weight_decay=wd,
        momentum=momentum)

    callbacks_constructed = []
    for name in callbacks:
        if name in avail_callbacks.__dict__:
            clbk = avail_callbacks.__dict__[name]()
            callbacks_constructed.append(clbk)

    training_loop(model=model,
        optimizer=optimizer,
        loss_function=blend_loss,
        metrics=[acc],
        train=train_loader, valid=valid, test=test,
        steps_per_epoch=len(train_loader),
        validation_steps=len(valid),
        test_steps=len(test),
        save_path=save_path,
        config=_CONFIG,
        custom_callbacks=callbacks_constructed
    )


def Greedy_main(args):
    ts = time.strftime('%Y_%m_%d %H:%M:%S', time.localtime())
    save_path = os.path.join(args.ckpt_path, f"{ts}_Greedy")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    gin_wrap(train, save_path, './models/Greedy/training_guided.gin')
