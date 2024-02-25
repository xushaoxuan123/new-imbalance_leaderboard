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
from models.MMCosine.MMCosine_main import train_epoch, valid
from train_model.support import ts_init, scalars_add, train_performance


def MMCosine_main(args):
    args.scaling = args.alpha  # scaling measn alpha
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

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=1e-4, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    train_dataset = AV_KS_Dataset(mode='train')
    test_dataset = AV_KS_Dataset(mode='test')
    val_dataset = AV_KS_Dataset(mode='val')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=16, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=16)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=16)


    if args.train:
        best_acc = -1
        if args.use_tensorboard:
            writer = ts_init(args)  # alpha means scaling factor

        # ts = time.strftime('%Y_%m_%d %H:%M:%S', time.localtime())
        # print(ts)
        # save_dir = os.path.join(args.ckpt_path, f"{ts}_{args.method}")
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # logger = get_logger("train_logger", logger_dir=save_dir)

        for epoch in range(args.epochs):
            batch_loss, batch_loss_a, batch_loss_v, acc, acc_a, acc_v = train_epoch(args, epoch, model, device,
                                                                 train_dataloader, optimizer, writer)

            scheduler.step()
            acc, acc_a, acc_v,valid_loss = valid(args, model, device, val_dataloader)

            if args.use_tensorboard:
                writer = scalars_add(writer, epoch, batch_loss, valid_loss, batch_loss_a, batch_loss_v, acc, acc_a, acc_v)

            if acc > best_acc:
                best_acc = train_performance(best_acc, acc_a, acc_v, batch_loss, valid_loss, args, acc, \
                                         epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),\
                                            {'alpha':args.alpha})

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