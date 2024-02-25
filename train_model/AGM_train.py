
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
from models.AGM.AGM_main import AGM_Config, train_epoch, test
from models.AGM.AGM_task import AGM_task, AGM_CONFIG
from train_model.support import ts_init, scalars_add, train_performance
def AGM_main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    cfgs = AGM_Config(args)
    device = torch.device('cuda:0')


    task = AGM_task(cfgs)
    train_dataloader = task.train_dataloader
    val_dataloader = task.valid_dataloader
    test_dataloader = task.test_dataloader
    
    model = task.model
    optimizer = task.optimizer
    scheduler = task.scheduler
    model.cuda(device)
    epoch_score_a = 0.
    epoch_score_v = 0.

    audio_lr_ratio = 1.0
    visual_lr_ratio = 1.0

    best_acc = 0.0001 # todo choose a reasonable value
    print('init best acc: ', best_acc)

    writer_path = os.path.join(cfgs.tensorboard_path)
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)

    if args.train :
        best_acc = 0.0
        if args.use_tensorboard:
            writer = ts_init(args)
        for epoch in range(cfgs.epochs):
            print('Epoch: {}'.format(epoch))
            epoch_loss, batch_loss_a, batch_loss_v, epoch_score_a, epoch_score_v = train_epoch(model, train_dataloader, optimizer, scheduler, cfgs, epoch, device, writer,
                                                epoch_score_a, epoch_score_v, audio_lr_ratio, visual_lr_ratio)



            test_acc, accuracy_a, accuracy_v, test_loss, validate_audio_batch_loss,\
                validate_visual_batch_loss = test(model, val_dataloader, cfgs, epoch, device, writer)

            if args.use_tensorboard:
                writer = scalars_add(writer, epoch, epoch_loss, test_loss, batch_loss_a, batch_loss_v, test_acc,accuracy_a, accuracy_v)

            best_acc = train_performance(best_acc, accuracy_a, accuracy_v, epoch_loss, test_loss, args, test_acc, model.state_dict(), optimizer.state_dict(),scheduler.state_dict(),AGM_CONFIG)
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

        acc, acc_a, acc_v, _ = test(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))
