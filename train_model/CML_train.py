import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.models import AVClassifier
from dataset.av_dataset import AV_KS_Dataset
from utils.utils import setup_seed
from models.CML.CML_main import train_epoch, valid
from train_model.support import ts_init, scalars_add, train_performance

def CML_main(args):
  gpu_ids = list(range(torch.cuda.device_count()))
  device = torch.device('cuda:0')

  if args.dataset == 'KineticSound':
    model = AVClassifier(args)
    train_dataset = AV_KS_Dataset(mode='train')
    test_dataset = AV_KS_Dataset(mode='test')
  model.to(device)
  model = torch.nn.DataParallel(model, device_ids=gpu_ids)
  model.cuda()


  # if args.dataset == 'KineticSound':
  #   train_dataset = AV_KS_Dataset(mode='train')
  #   test_dataset = AV_KS_Dataset(mode='test')
  
  train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=16,pin_memory=True)
  test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=16)

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

      batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device, train_dataloader, optimizer)
      scheduler.step()
      acc, acc_a, acc_v, val_loss = valid(args, model, device, test_dataloader)
      if args.use_tensorboard:
        writer = scalars_add(writer, epoch, batch_loss, val_loss, batch_loss_a, batch_loss_v, acc, acc_a, acc_v)
      best_acc = train_performance(best_acc, acc_a, acc_v, batch_loss, val_loss, args, acc, epoch, model.state_dict(),optimizer.state_dict(),scheduler.state_dict(),{'alpha':args.alpha})
    writer.close()

