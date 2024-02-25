import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.models import AVClassifier
from utils.utils import setup_seed, weight_init
from models.CKF.CKF_main import train_epoch, valid
from train_model.support import ts_init, scalars_add, train_performance, Optimizer_build, Dataloader_build

def CKF_main(args):
  # gpu_ids = list(range(torch.cuda.device_count()))
  setup_seed(args.random_seed)
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
  device = torch.device('cuda:0')
  model = AVClassifier(args)
  model.apply(weight_init)
  train_dataloader, test_dataloader, val_dataloader = Dataloader_build(args)
  optimizer, optimizer_alpha, scheduler = Optimizer_build(args, model)


  if args.train:
    best_acc = 0.0
    if args.use_tensorboard:
       writer = ts_init(args)
    for epoch in range(args.epochs):

      print('Epoch: {}: '.format(epoch))

      batch_loss, batch_loss_a, batch_loss_v,_ , _, _ = train_epoch(args,model, device, optimizer,train_dataloader, optimizer_alpha)
      scheduler.step()
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

