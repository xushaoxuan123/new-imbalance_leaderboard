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
from models.PMR.PMR_main import train_epoch, valid, calculate_prototype
from train_model.support import ts_init, scalars_add, train_performance

def PMR_main(args):
    args.momentum_coef = 0.5
    args.embed_dim = 512
    device = torch.device('cuda:0')

    model = AVClassifier(args)
    model.apply(weight_init)
    model.to(device)

    # model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)
    elif args.optimizer == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
        scheduler = None
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
        scheduler = None

    train_dataset = AV_KS_Dataset(mode='train')
    val_dataset = AV_KS_Dataset(mode='val')
    test_dataset = AV_KS_Dataset(mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, pin_memory=False)


    if args.train:
        # tensorboard
        best_acc = 0
        epoch = 0
        audio_proto, visual_proto = calculate_prototype(args, model, train_dataloader, device, epoch)
        if args.use_tensorboard:
            writer = ts_init()
        for epoch in range(args.epochs):
            batch_loss, batch_loss_a, batch_loss_v, batch_loss_a_p, batch_loss_v_p, a_angle, v_angle, ratio_a, ratio_a_p, \
               a_diff, v_diff = train_epoch(args, epoch, model, device, train_dataloader, optimizer, scheduler,
                              audio_proto, visual_proto,)
            audio_proto, visual_proto = calculate_prototype(args, model, train_dataloader, device, epoch, audio_proto, visual_proto)
            # print('proto22', audio_proto[22], visual_proto[22])
            acc, acc_a, acc_v, acc_a_p, acc_v_p, valid_loss = valid(args, model, device, val_dataloader, audio_proto, visual_proto, epoch, writer)
            # logger.info('epoch: ', epoch, 'loss: ', batch_loss, batch_loss_a_p, batch_loss_v_p)
            # logger.info('epoch: ', epoch, 'acc: ', acc, 'acc_v_p: ', acc_v_p, 'acc_a_p: ', acc_a_p)
            if args.use_tensorboard:
                writer = scalars_add(writer, epoch, batch_loss, valid_loss, batch_loss_a, batch_loss_v, acc, acc_a, acc_v)
            best_acc = train_performance(best_acc, acc_a, acc_v, batch_loss, valid_loss, args, acc, \
                                         epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),\
                                            {'alpha':args.alpha, 'embed_dim':args.embed_dim, 'coef':args.momentum_coef})
            

    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model.load_state_dict(state_dict)
        print('Trained model loaded!')

        acc, acc_a, acc_v, acc_vp, acc_ap, _loss = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))
