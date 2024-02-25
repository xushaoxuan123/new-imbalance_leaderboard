import time

from utils.utils import setup_seed
from dataset.av_dataset import AV_KS_Dataset
# from transformers import get_cosine_schedule_with_warmup
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models.models import AVClassifier
from sklearn import metrics
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import itertools
# from min_norm_solvers import MinNormSolver
import numpy as np
from tqdm import tqdm
import argparse
import re
import pickle
import os
from operator import mod
from train_model.AGM_train import AGM_main
from models.Greedy.train import Greedy_main
from train_model.MMCosine_train import MMCosine_main
# from models.PMR.PMR_main import PMR_main
from train_model.PMR_train import PMR_main
# from models.CML.CML_main import CML_train_epoch, CML_valid
from train_model.OGM_train import OGM_main
from train_model.AMCo_train import ACMo_main
from train_model.GBlending_train import GBleding_main
from utils.utils import get_logger
from train_model.CML_train import CML_main


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='KineticSound',
                        help='KineticSound, CREMAD, K400, VGGSound, Audioset, VGGPart, UCF101')
    parser.add_argument('--model', default='model', type=str)
    parser.add_argument('--fusion_method', default='concat', type=str)
    parser.add_argument('--method', required=True, default='AGM', type=str, help='AGM, OGM, Greedy...')
    parser.add_argument('--n_classes', default=31, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=60, type=int)

    parser.add_argument('--optimizer', default='sgd',
                        type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1,
                        type=float, help='decay coefficient')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

    parser.add_argument('--modulation_starts', default=10, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=60, type=int, help='where modulation ends')
    parser.add_argument('--alpha', default=1.0, type=float, help='alpha in OGM-GE, MSBD')

    parser.add_argument('--ckpt_path', default='/data/users/shaoxuan_xu/results/leaderboard/Models/ckpt/',
                        type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_false', help='turn on train mode')  # default True
    parser.add_argument('--clip_grad', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--use_tensorboard', default=True,
                        type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', default='/data/users/shaoxuan_xu/results/leaderboard/log/',
                        type=str, help='path to save tensorboard logs')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0',
                        type=str, help='GPU ids')
    ##GBlending
    parser.add_argument('--super_epoch', default=10, type=int, help='the value of super_epoch')

    ##AMCo
    parser.add_argument('--U', default='100',
                        type=int, help='ACMo_U')##### new
    parser.add_argument('--eps', default='0.3',
                        type=float, help='ACMo_eps')##### new
    parser.add_argument('--sigma', default='0.5',
                        type=float, help='ACMo_sigma')##### new
    
    ##CML
    parser.add_argument('--lam', default='0.5',
                        type=float, help='CML_lambda')##### new
    

    ##MMcosine
    parser.add_argument('--temperature', default='0.1',
                        type=float, help='MMcosine_temp')##### new
    parser.add_argument('--EPISILON', default='1e-10',
                        type=float, help='MMcosine_EPISILON')##### new
    parser.add_argument('--scaling',default=10,type=float,help='scaling parameter in mmCosine')

    ## UNM
    parser.add_argument('--lam_dill',default = 50, type = float, help = 'the parameter for loss')
    parser.add_argument('--lam_task',default = 1, type = float, help = 'the parameter for loss_a and loss_v')
    return parser.parse_args()



def main():
    args = get_arguments()
    print(args)
    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0')

    if args.method == 'AGM':
        AGM_main(args)
    if args.method == 'Greedy':
        Greedy_main(args)
    if args.method == 'MMCosine':
        MMCosine_main(args)
    if args.method == 'PMR':
        PMR_main(args)
    if args.method == 'OGM':#OGM method
        OGM_main(args)
    if args.method == 'ACMo':
        ACMo_main(args)
    if args.method == 'GBlending':
        GBleding_main(args)
    if args.method == 'CML':
        CML_main(args)
    if args.method == 'CKF':
        from train_model.CKF_train import CKF_main
        CKF_main(args)
    if args.method == 'MSBD':
        from train_model.MSBD_train import MSBD_main
        MSBD_main(args)
    if args.method == 'audio':
        from models.Uni_model.AV_train import train_main
        train_main(args,args.method)
    if args.method == 'visual':
        from models.Uni_model.AV_train import train_main
        train_main(args,args.method)
    if args.method == 'UNM':
        from train_model.UNM_train import UNM_main

        UNM_main(args)

    # else:
        # # logger name
        # ts = time.strftime('%Y_%m_%d %H:%M:%S', time.localtime())
        # print(ts)
        # save_dir = os.path.join(args.ckpt_path, f"{ts}_{args.method}")
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # logger = get_logger("train_logger", logger_dir=save_dir)
        # # writer path
        # writer_path = os.path.join(args.tensorboard_path)
        # if not os.path.exists(writer_path):
        #     os.mkdir(writer_path)
        # log_name = 'model_{}_{}_{}_epoch{}_batch{}_lr{}_alpha{}'.format(
        #     args.method, args.optimizer, args.dataset, args.epochs, args.batch_size, args.learning_rate, args.alpha)
        # writer = SummaryWriter(os.path.join(writer_path, log_name))


        # model = AVClassifier(args)
        # model.to(device)
        # # model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        # # model.cuda()
        # train_dataset = AV_KS_Dataset(mode='train')
        # test_dataset = AV_KS_Dataset(mode='test')
        # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
        #                               shuffle=True, num_workers=16, pin_memory=False)
        # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
        #                              shuffle=False, num_workers=16)

        # if args.optimizer == 'sgd':
        #     optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        # elif args.optimizer == 'adam':
        #     optimizer = optim.Adam(
        #         model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

        # print(len(train_dataloader))
        # # here begins
        # if args.train:
        #     best_acc = 0
        #     for epoch in range(args.epochs):
        #         print('Epoch: {}: '.format(epoch))

        #         if args.method == 'CML':
        #             batch_loss, l_mm, l_a, l_v = CML_train_epoch(args, epoch, model, device, train_dataloader, optimizer, scheduler, writer, logger)
        #             acc, acc_a, acc_v, vloss = CML_valid(args, model, device, test_dataloader)
        #         else:
        #             batch_loss,l_mm,l_a,l_v,cos_a,cos_v = train_epoch(
        #             args, epoch, model, device, train_dataloader, optimizer, scheduler, writer)
        #             acc, acc_a, acc_v, vloss = valid(args, model, device, test_dataloader)

        #         # if args.dataset == 'Audioset':
        #         #     mAP, mAUC, acc, vloss = valid(
        #         #         args, model, device, test_dataloader)
        #         # else:
        #         #     acc, acc_a,acc_v, vloss = valid(args, model, device, test_dataloader)

        #         writer.add_scalars('Loss', {'Total Loss': batch_loss}, epoch)
        #         writer.add_scalars(
        #             'Evaluation', {'Total Accuracy': acc,
        #                            'audio acc': acc_a,
        #                            'visual acc': acc_v}, epoch)

        #         if acc > best_acc:
        #             best_acc = float(acc)
        #             if not os.path.exists(args.ckpt_path):
        #                 os.mkdir(args.ckpt_path)
        #             # model path
        #             model_name = 'best_model_{}_of_{}_epoch{}_batch{}_lr{}_alpha{}.pth'.format(
        #                 args.method, args.optimizer, args.epochs, args.batch_size, args.learning_rate,
        #                 args.alpha)

        #             saved_dict = {'saved_epoch': epoch,
        #                           'acc': acc,
        #                           'model': model.state_dict(),
        #                           'optimizer': optimizer.state_dict(),
        #                           'scheduler': scheduler.state_dict()}

        #             save_dir = os.path.join(args.ckpt_path, model_name)
        #             torch.save(saved_dict, save_dir)
        #             logger.info('The best model has been saved at {}.'.format(save_dir))
        #             logger.info("Loss: {:.4f}, Acc: {:.4f}, Acc_a: {:.4f}, Acc_v: {:.4f}, vloss: {:.4f}".format(
        #                 batch_loss, acc, acc_a, acc_v, vloss))
        #         else:
        #             logger.info(
        #                 "Loss: {:.4f}, Acc: {:.4f}, Acc_a: {:.4f}, Acc_v: {:.4f},Best Acc: {:.4f}, vloss: {:.4f}".format(
        #                     batch_loss, acc, acc_a, acc_v, best_acc, vloss))

        # else:
        #     pass


if __name__ == "__main__":
    main()
