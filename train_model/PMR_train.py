import argparse
import os
from time import time
import numpy as np
import torch
from models.models import AVClassifier
from utils.utils import setup_seed, weight_init
from models.PMR.PMR_main import train_epoch, valid, calculate_prototype
from train_model.support import ts_init, scalars_add, train_performance, Dataloader_build, Optimizer_build

def PMR_main(args):
    args.momentum_coef = 0.5
    args.embed_dim = 512
    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device('cuda:0')
    model = AVClassifier(args)
    model.apply(weight_init)
    model.to(device)

    # model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    optimizer, scheduler = Optimizer_build(args, model)

    train_dataloader, test_dataloader, val_dataloader = Dataloader_build(args)


    if args.train:
        # tensorboard
        best_acc = 0
        epoch = 0
        audio_proto, visual_proto = calculate_prototype(args, model, train_dataloader, device, epoch)
        if args.use_tensorboard:
            writer = ts_init(args)
        for epoch in range(args.epochs):
            batch_loss, batch_loss_a, batch_loss_v, batch_loss_a_p, batch_loss_v_p, a_angle, v_angle, ratio_a, ratio_a_p, \
               a_diff, v_diff = train_epoch(args, epoch, model, device, train_dataloader, optimizer, scheduler,
                              audio_proto, visual_proto,)
            audio_proto, visual_proto = calculate_prototype(args, model, train_dataloader, device, epoch, audio_proto, visual_proto)
            # print('proto22', audio_proto[22], visual_proto[22])
            acc, acc_a, acc_v, acc_a_p, acc_v_p, valid_loss = valid(args, model, device, val_dataloader, audio_proto, visual_proto, epoch)
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
