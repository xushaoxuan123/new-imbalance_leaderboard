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
from utils.utils import setup_seed, weight_init
from models.MMCosine.MMCosine_main import train_epoch, valid
from train_model.support import ts_init, scalars_add, train_performance, Dataloader_build, Optimizer_build

def MMCosine_main(args):
    args.lam = 0 # 不用蒸馏 for fair
    print(args)
    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0')
    model = AVClassifier(args)
    model.to(device)
    # model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda()
    # todo
    # if args.audio_pretrain != 'None':
    #     loaded_dict_audio = torch.load(args.audio_pretrain)
    #     state_dict_audio = loaded_dict_audio
    #     model.module.audio_net.load_state_dict(state_dict_audio, strict=False)
    # if args.visual_pretrain != 'None':
    #     loaded_dict_visual = torch.load(args.visual_pretrain)
    #     state_dict_visual = loaded_dict_visual
    #     model.module.visual_net.load_state_dict(state_dict_visual, strict=False)


    optimizer, scheduler = Optimizer_build(args, model)

    train_dataloader, test_dataloader, val_dataloader = Dataloader_build(args)


    if args.train:
        best_acc = 0.0
        if args.use_tensorboard:
            writer = ts_init(args)
        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            batch_loss, batch_loss_a, batch_loss_v,_ , _, _ = train_epoch(args, epoch, model, device, train_dataloader, optimizer, writer)
            scheduler.step()
            acc, acc_a, acc_v, val_loss = valid(args, model, device, test_dataloader)
            if args.use_tensorboard:
                writer = scalars_add(writer, epoch, batch_loss, val_loss, batch_loss_a, batch_loss_v, acc, acc_a, acc_v)
            best_acc = train_performance(best_acc, acc_a, acc_v, batch_loss, val_loss, args, acc, epoch, model.state_dict(),optimizer.state_dict(),scheduler.state_dict(),{'scaling':args.scaling})
            writer.close()
        else:
            # first load trained model
            loaded_dict = torch.load(args.ckpt_path)
            # epoch = loaded_dict['saved_epoch']
            fusion = loaded_dict['fusion']
            state_dict = loaded_dict['model']
            # optimizer_dict = loaded_dict['optimizer']
            # scheduler = loaded_dict['scheduler']

            assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

            model = model.load_state_dict(state_dict)
            print('Trained model loaded!')

            acc, acc_a, acc_v = valid(args, model, device, test_dataloader, epoch=1001)
            print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))