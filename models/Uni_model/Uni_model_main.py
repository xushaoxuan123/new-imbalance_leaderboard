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
from models.models import AVClassifier , AClassifier, VClassifier# , CGClassifier
from utils.utils import setup_seed, weight_init, get_logger
# from dataset.VGGSoundDataset import VGGSound
from train_model.support import ts_init, train_performance, scalars_add
import time


def train_epoch(args, epoch, model, model_a, model_v, device, optimizer, dataloader):
    criterion = nn.CrossEntropyLoss()
    criterion_MSE = nn.MSELoss()
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


        out, a, v = model(spec.unsqueeze(1).float(), image.float())
        a_pre, _ = model_a(spec.unsqueeze(1).float())
        v_pre, _ = model_v(image.float())

        out_v = (torch.mm(v, torch.transpose(model.head.weight[:,512:], 0, 1)) +
                    model.head.bias/2)
        out_a = (torch.mm(a, torch.transpose(model.head.weight[:,:512], 0, 1)) +
                    model.head.bias/2)
        loss = criterion(out, label)
        if args.modulation_starts <= epoch <= args.modulation_ends:
            loss_aa = criterion_MSE(out_a, a_pre)
            loss_vv = criterion_MSE(out_v, v_pre)
            loss = loss * args.lam_task + (loss_aa + loss_vv)* args.lam_dill
        prediction = softmax(out)
        pred_v = softmax(out_v)
        pred_a = softmax(out_a)
        
        _loss += loss.item()
        loss.backward()
        optimizer.step()
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
                out, a, v = model(spec.unsqueeze(1).float(), image.float())
            else:
                a, v, out = model(spec, image)  # gray colored

            out_v = (torch.mm(v, torch.transpose(model.head.weight[:,512:], 0, 1)) +
                    model.head.bias/2)
            out_a = (torch.mm(a, torch.transpose(model.head.weight[:,:512], 0, 1)) +
                    model.head.bias/2)
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




def UNM_main(args):
   # gpu_ids = list(range(torch.cuda.device_count()))
    print(1)
    device = torch.device('cuda:0')
    model = AVClassifier(args)
    load_dict = torch.load("/data/users/shaoxuan_xu/results/leaderboard/Models/ckpt/audio_best_model_of_KineticSound_opti_sgd_batch_64_lr_0.001_alpha_1.0_.pth")
    state_dict = load_dict['model']
    model_a = AClassifier(args)
    model_a.load_state_dict(state_dict)
    load_dict = torch.load("/data/users/shaoxuan_xu/results/leaderboard/Models/ckpt/visual_best_model_of_KineticSound_opti_sgd_batch_64_lr_0.001_alpha_1.0_.pth")
    state_dict = load_dict['model']
    model_v = VClassifier(args)
    model_v.load_state_dict(state_dict)

    model.apply(weight_init)
    if args.dataset == 'KineticSound':
        train_dataset = AV_KS_Dataset(mode='train')
        test_dataset = AV_KS_Dataset(mode='val')
    model.to(device)
    model_a.to(device)
    model_v.to(device)
    # model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    
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

            batch_loss, batch_loss_a, batch_loss_v,_ , _, _ = train_epoch(args, epoch, model, model_a, model_v, device, optimizer, train_dataloader)
            scheduler.step()
            acc, acc_a, acc_v, val_loss = valid(args, model, device, test_dataloader)
            if args.use_tensorboard:
                writer = scalars_add(writer, epoch, batch_loss, val_loss, batch_loss_a, batch_loss_v, acc, acc_a, acc_v)
            best_acc = train_performance(best_acc, acc_a, acc_v, batch_loss, val_loss, args, acc, epoch, model.state_dict(),optimizer.state_dict(),scheduler.state_dict(),{'alpha':args.alpha})
        writer.close()
