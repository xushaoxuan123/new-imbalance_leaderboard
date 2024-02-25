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
from models.models import AClassifier, VClassifier # , CGClassifier
from utils.utils import setup_seed, weight_init, get_logger
# from dataset.VGGSoundDataset import VGGSound
from train_model.support import ts_init, train_performance, scalars_add
import time


def train_epoch(args, model, device, optimizer, dataloader, mode):
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
        optimizer.zero_grad()
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        if args.dataset != 'CGMNIST':
            if mode == 'audio':
                out, f = model(spec.unsqueeze(1).float())
            else :
                out, f = model(image.float())

        loss = criterion(out, label)
        _loss += loss.item()
        prediction = softmax(out)

        loss.backward()
        optimizer.step()

    return _loss / len(dataloader)
    

def valid(args, model, device, dataloader, mode):
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
    with torch.no_grad():
        for step, (image, spec, label) in enumerate(dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

        if args.dataset != 'CGMNIST':
            if mode == 'audio':
                out, f = model(spec.unsqueeze(1).float())
            else :
                out, f = model(image.float())

            loss = criterion(out, label)
            _loss += loss.item()
            prediction = softmax(out)


            for j in range(image.shape[0]):
                ma = np.argmax(prediction[j].cpu().data.numpy())
                num[label[j]] += 1.0

                if np.asarray(label[j].cpu()) == ma:
                    acc[label[j]] += 1.0

    return sum(acc) / sum(num), _loss/sum(num)




def train_main(args, mode):
    # gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0')
    if mode == 'audio':
        model = AClassifier(args)
    else :
        model = VClassifier(args)
    model.apply(weight_init)
    if args.dataset == 'KineticSound':
        train_dataset = AV_KS_Dataset(mode='train')
        test_dataset = AV_KS_Dataset(mode='val')
    model.to(device)
    # model = torch.nn.DataParallel(model, device_ids=gpu_ids)



    # if args.dataset == 'KineticSound':
    #   train_dataset = AV_KS_Dataset(mode='train')
    #   test_dataset = AV_KS_Dataset(mode='test')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                    shuffle=True,pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                    shuffle=False)

    # params = [
    #   {"params": model.module.audio_net.parameters(), "lr": args.learning_rate * args.encoder_lr_decay},
    #   {"params": model.module.visual_net.parameters(), "lr": args.learning_rate * args.encoder_lr_decay},
    #   {"params": model.module.fusion_module.parameters(), "lr": args.learning_rate},
    # ]

    if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)

    elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)

    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.train:
        best_acc = 0.0
        if args.use_tensorboard:
            writer = ts_init(args)
        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            batch_loss = train_epoch(args, model, device, optimizer,train_dataloader, args.method)
            scheduler.step()
            acc, val_loss = valid(args, model, device, test_dataloader,args.method)
            if args.use_tensorboard:
                writer = scalars_add(writer, epoch, batch_loss, val_loss, 0, 0, acc, 0, 0)
            best_acc = train_performance(best_acc, 0, 0, batch_loss, val_loss, args, acc, epoch, model.state_dict(),optimizer.state_dict(),scheduler.state_dict(),{'alpha':args.alpha})
            writer.close()

    