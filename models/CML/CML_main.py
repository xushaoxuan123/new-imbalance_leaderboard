import time

from utils.utils import setup_seed
from dataset.av_dataset import AV_KS_Dataset
# from transformers import get_cosine_schedule_with_warmup
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models.models import AVClassifier
from sklearn import metrics
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import itertools
# from min_norm_solvers import MinNormSolver
import numpy as np
from tqdm import tqdm
import argparse

def conf_loss(conf, pred, conf_x, pred_x, label):
    # print(conf.shape, pred.shape, conf_x.shape, pred_x.shape, label.shape)
    # sign==1 => ( pred false || pred_x true)
    # sign == 0 => pred true and prex , 此时loss取0
    # sign = (~((pred == label) & (pred_x != label))).long()  # trick 1
    # print(sign)
    return torch.nn.ReLU()(torch.sub(conf_x, conf) * sign).sum(), sign.sum()



def CML_train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None, logger=None):
    criterion = nn.CrossEntropyLoss()
    model.train()
    n_classes = 31
    print("Start training ... ")
    weight_size = model.head.weight.size(1)

    _loss = 0
    num = [0.0 for _ in range(n_classes)]
    acc = [0.0 for _ in range(n_classes)]
    acc_a = [0.0 for _ in range(n_classes)]
    acc_v = [0.0 for _ in range(n_classes)]

    loss_value_mm = []
    loss_value_a = []
    loss_value_v = []
    conf_loss_hit_a = 0
    conf_loss_hit_v = 0

    for step, (images, spec, label) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        images = images.to(device)
        spec = spec.to(device)
        label = label.to(device)
        out, out_a, out_v = model(spec.unsqueeze(1).float(), images.float())

        loss_mm = criterion(out, label)
        # loss_a = criterion(out_a, label)
        # loss_v = criterion(out_v, label)
        prediction = F.softmax(out, dim=1)
        out_a = (torch.mm(out_a, torch.transpose(model.head.weight[:, :weight_size // 2], 0, 1)) +
                 model.head.bias / 2)
        out_v = (torch.mm(out_v, torch.transpose(model.head.weight[:, weight_size // 2:], 0, 1)) +
                 model.head.bias/2)

        pred_a = F.softmax(out_a, dim=1)
        pred_v = F.softmax(out_v, dim=1)

        loss = loss_mm
        if args.modulation_starts <= epoch <= args.modulation_ends:
            conf, pred = torch.max(prediction, dim=1)
            conf_a, pred_a = torch.max(pred_a, dim=1)
            conf_v, pred_v = torch.max(pred_v, dim=1)
            loss_a, count = conf_loss(conf, pred, conf_a, pred_a, label)
            conf_loss_hit_a += count
            loss += loss_a
            loss_v, count = conf_loss(conf, pred, conf_v, pred_v, label)
            conf_loss_hit_v += count
            loss += loss_v

        loss.backward()

        optimizer.step()
        scheduler.step()
        _loss += loss.item()

        # acc
        for j in range(images.shape[0]):
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

    accuracy = sum(acc) / sum(num)
    accuracy_a = sum(acc_a) / sum(num)
    accuracy_v = sum(acc_v) / sum(num)
    conf_hit_ratio_a = conf_loss_hit_a / sum(num)
    conf_hit_ratio_v = conf_loss_hit_v / sum(num)
    writer.add_scalars('Epoch Accuracy(train)', {'accuracy': accuracy,
                                                 'accuracy audio': accuracy_a,
                                                 'accuracy visual': accuracy_v}, epoch)
    writer.add_scalars('Epoch conf hit ratio', {
                                                 'a conf hit ratio': conf_hit_ratio_a,
                                                 'v conf hit ratio': conf_hit_ratio_v}, epoch)

    logger.info(f'conf loss hit ratio audio: {conf_hit_ratio_a}, visual: {conf_hit_ratio_v}')

    return _loss / len(dataloader), loss_value_mm, loss_value_a, loss_value_v


def CML_valid(args, model, device, dataloader):
    n_classes = 31
    cri = nn.CrossEntropyLoss()
    # cri = one_hot_CrossEntropy()
    _loss = 0
    weight_size = model.head.weight.size(1)

    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (images, spec, label) in tqdm(enumerate(dataloader)):
            spec = spec.to(device)
            images = images.to(device)
            label = label.to(device)

            out, out_a, out_v = model(spec.unsqueeze(1).float(), images.float())

            # loss_a = criterion(out_a, label)
            # loss_v = criterion(out_v, label)
            prediction = F.softmax(out, dim=1)
            loss = cri(out, label)

            out_a = (torch.mm(out_a, torch.transpose(model.head.weight[:, :weight_size // 2], 0, 1)) +
                     model.head.bias / 2)
            out_v = (torch.mm(out_v, torch.transpose(model.head.weight[:, weight_size // 2:], 0, 1)) +
                     model.head.bias / 2)

            _loss += loss.item()

            for i, item in enumerate(label):

                ma = prediction[i].cpu().data.numpy()
                index_ma = np.argmax(ma)
                # print(index_ma, label_index)
                num[label[i]] += 1.0
                if index_ma == label[i]:
                    acc[label[i]] += 1.0

                ma_audio = out_a[i].cpu().data.numpy()
                index_ma_audio = np.argmax(ma_audio)
                if index_ma_audio == label[i]:
                    acc_a[label[i]] += 1.0

                ma_visual = out_v[i].cpu().data.numpy()
                index_ma_visual = np.argmax(ma_visual)
                if index_ma_visual == label[i]:
                    acc_v[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num), _loss / len(dataloader)