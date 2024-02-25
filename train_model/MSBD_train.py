import argparse
import os
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter, get_logger
import pdb

from models.OGM.OGM_CD import CramedDataset
from dataset.av_dataset import AV_KS_Dataset
from models.OGM.OGM_AVC import AVClassifier
from utils.utils import setup_seed, weight_init
from models.MSBD.MSBD_main import train_epoch, valid
from train_model.support import ts_init, scalars_add, train_performance, Dataloader_build, Optimizer_build

def MSBD_main(args):
    print(args)
    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0')
    model = AVClassifier(args)
    model.to(device)
    # model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda()

    optimizer, scheduler = Optimizer_build(args, model)

    train_dataloader, test_dataloader, val_dataloader = Dataloader_build(args)


    if args.train:
        best_acc = 0.0
        if args.use_tensorboard:
            writer = ts_init(args)
        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            batch_loss, batch_loss_a, batch_loss_v,_ , _, _ = train_epoch(args, epoch, model, device, train_dataloader, optimizer)
            scheduler.step()
            acc, acc_a, acc_v, val_loss = valid(args, model, device, test_dataloader)
            if args.use_tensorboard:
                writer = scalars_add(writer, epoch, batch_loss, val_loss, batch_loss_a, batch_loss_v, acc, acc_a, acc_v)
            best_acc = train_performance(best_acc, acc_a, acc_v, batch_loss, val_loss, args, acc, epoch, model.state_dict(),optimizer.state_dict(),scheduler.state_dict(),{'alpha':args.alpha})
        writer.close()