import argparse
import os
from time import time
import numpy as np
import torch
from models.models import AVClassifier, AClassifier, VClassifier
from utils.utils import setup_seed, weight_init
from models.Uni_model.Uni_model_main import train_epoch, valid
from train_model.support import ts_init, scalars_add, train_performance, Dataloader_build, Optimizer_build
def UNM_main(args):
  # gpu_ids = list(range(torch.cuda.device_count()))
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
  train_dataloader,test_dataloader,valid_dataloader = Dataloader_build(args)
  model.to(device)
  model_a.to(device)
  model_v.to(device)
  # model = torch.nn.DataParallel(model, device_ids=gpu_ids)

  # params = [
  #   {"params": model.module.audio_net.parameters(), "lr": args.learning_rate * args.encoder_lr_decay},
  #   {"params": model.module.visual_net.parameters(), "lr": args.learning_rate * args.encoder_lr_decay},
  #   {"params": model.module.fusion_module.parameters(), "lr": args.learning_rate},
  # ]

  optimizer, scheduler = Optimizer_build(args, model)


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
