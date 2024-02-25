import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.models import AVClassifier_gb, RFClassifier_gb
from dataset.av_dataset import AV_KS_Dataset
from utils.utils import setup_seed
from models.GBlending.GBlending_main import super_epoch_origin, train_epoch, valid
from train_model.support import ts_init, scalars_add, train_performance
def GBleding_main(args):
  cona_all = []
  conv_all = []

  gpu_ids = list(range(torch.cuda.device_count()))

  device = torch.device('cuda:0')

  if args.dataset == 'KineticSound':
    model = AVClassifier_gb(args)
    temp_model = AVClassifier_gb(args)
    train_dataset = AV_KS_Dataset(mode='train')
    test_dataset = AV_KS_Dataset(mode='test')
  elif args.dataset == 'UCF-101':
    model = RFClassifier_gb(args)
    temp_model = RFClassifier_gb(args)
  model.to(device)
  model = torch.nn.DataParallel(model, device_ids=gpu_ids)
  model.cuda()

  temp_model.to(device)
  temp_model = torch.nn.DataParallel(temp_model, device_ids=gpu_ids)
  temp_model.cuda()

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
        temp_optimizer = optim.SGD(temp_model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
  elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
        temp_optimizer = optim.Adam(temp_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)

  scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)
  temp_scheduler = optim.lr_scheduler.StepLR(temp_optimizer, args.lr_decay_step, args.lr_decay_ratio)

  if args.train:
    best_acc = 0.0
    if args.use_tensorboard:
       writer = ts_init(args)
    for epoch in range(args.epochs):

      print('Epoch: {}: '.format(epoch))
      
      if epoch % args.super_epoch == 0:
        wa, wv, wav = super_epoch_origin(args, model, temp_model, device, train_dataloader, test_dataloader, temp_optimizer)

      batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device, train_dataloader, optimizer, wa, wv, wav, writer)
      scheduler.step()
      temp_scheduler.step()
      acc, acc_a, acc_v, val_loss = valid(args, model, device, test_dataloader)
      if args.use_tensorboard:
        writer = scalars_add(writer, epoch, batch_loss, val_loss, batch_loss_a, batch_loss_v, acc, acc_a, acc_v)
      best_acc = train_performance(best_acc, acc_a, acc_v, batch_loss, val_loss, args, acc, epoch, model.state_dict(),optimizer.state_dict(),scheduler.state_dict(),{'alpha':args.alpha})
    writer.close()
  else:
    # first load trained model
    loaded_dict = torch.load(args.ckpt_path)
    # epoch = loaded_dict['saved_epoch']
    modulation = loaded_dict['fusion_method']
    # alpha = loaded_dict['alpha']
    fusion = loaded_dict['fusion']
    state_dict = loaded_dict['model']
    # optimizer_dict = loaded_dict['optimizer']
    # scheduler = loaded_dict['scheduler']

    # assert modulation == args.fusion_method, 'inconsistency between modulation method of loaded model and args !'
    # assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

    model = model.load_state_dict(state_dict)
    print('Trained model loaded!')

    acc, acc_a, acc_v, _ = valid(args, model, device, test_dataloader)
    print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))
