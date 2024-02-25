import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.av_dataset import AV_KS_Dataset
# from dataset.CGMNIST import CGMNISTDataset
# from dataset.CramedDataset import CramedDataset
# from dataset.AVEDataset import AVEDataset
# from dataset.dataset import AVDataset
from models.models import AVClassifier # , CGClassifier
from utils.utils import setup_seed, weight_init, get_logger
# from dataset.VGGSoundDataset import VGGSound
import time


def train_epoch(args, model, device, optimizer, dataloader, optimizer_alpha):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    _loss = 0
    _loss_a = 0
    _loss_v = 0
    n_classes = 31
    num = [0.0 for _ in range(n_classes)]
    acc = [0.0 for _ in range(n_classes)]
    acc_a = [0.0 for _ in range(n_classes)]
    acc_v = [0.0 for _ in range(n_classes)]

    for step, (image, spec, label) in enumerate(dataloader):
        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        optimizer_alpha.zero_grad()
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        if args.dataset != 'CGMNIST':
            a, v, out = model(spec.unsqueeze(1).float(), image.float())
        else:
            a, v, out = model(spec, image)  # gray colored

        out_v = (torch.mm(v, torch.transpose(model.head_video.weight, 0, 1)) +
                    model.head_video.bias)
        out_a = (torch.mm(a, torch.transpose(model.head_audio.weight, 0, 1)) +
                    model.head_audio.bias)
        loss = criterion(out, label)
        _loss += loss.item()
        prediction = softmax(out)
        pred_v = softmax(out_v)
        pred_a = softmax(out_a)
        loss.backward()
        optimizer.step()
        optimizer_alpha.step()
        loss_a = criterion(out_a,label)
        loss_v = criterion(out_v,label)
        _loss_a += loss_a
        _loss_v += loss_v

        for j in range(image.shape[0]):
            ma = np.argmax(prediction[j].cpu().data.numpy())
            v = np.argmax(pred_v[j].cpu().data.numpy())
            a = np.argmax(pred_a[j].cpu().data.numpy())
            num[label[j]] += 1.0

            if np.asarray(label[j].cpu()) == ma:
                acc[label[j]] += 1.0
            if np.asarray(label[j].cpu()) == v:
                acc_v[label[j]] += 1.0
            if np.asarray(label[j].cpu()) == a:
                acc_a[label[j]] += 1.0
    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), \
        sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)
    
        
def CKF_optim(model, optimizer, loss):
    model.alpha.requires_grad = True
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.alpha.requires_grad = False
    print('alpha is : {:0}'.format(0))

def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    n_classes = 31
    total_batch = len(dataloader)
    start_time = time.time()
    _loss = 0.
    _loss_a = 0
    _loss_v = 0
    num = [0.0 for _ in range(n_classes)]
    acc = [0.0 for _ in range(n_classes)]
    acc_a = [0.0 for _ in range(n_classes)]
    acc_v = [0.0 for _ in range(n_classes)]
    with torch.no_grad():
        for step, (image, spec, label) in enumerate(dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            if args.dataset != 'CGMNIST':
                a, v, out = model(spec.unsqueeze(1).float(), image.float())
            else:
                a, v, out = model(spec, image)  # gray colored

            out_v = (torch.mm(v, torch.transpose(model.head_video.weight, 0, 1)) +
                        model.head_video.bias)
            out_a = (torch.mm(a, torch.transpose(model.head_audio.weight, 0, 1)) +
                        model.head_audio.bias)
            loss = criterion(out, label)
            _loss += loss.item()
            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for j in range(image.shape[0]):
                ma = np.argmax(prediction[j].cpu().data.numpy())
                v = np.argmax(pred_v[j].cpu().data.numpy())
                a = np.argmax(pred_a[j].cpu().data.numpy())
                num[label[j]] += 1.0

                if np.asarray(label[j].cpu()) == ma:
                    acc[label[j]] += 1.0
                if np.asarray(label[j].cpu()) == v:
                    acc_v[label[j]] += 1.0
                if np.asarray(label[j].cpu()) == a:
                    acc_a[label[j]] += 1.0
    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num), _loss/sum(num)

