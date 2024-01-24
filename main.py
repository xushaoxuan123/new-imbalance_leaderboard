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
from models.AGM.AGM_main import AGM_main
from models.Greedy.train import Greedy_main
from models.MMCosine.MMCosine_main import MMCosine_main
from models.PMR.PMR_main import PMR_main
from models.CML.CML_main import CML_train_epoch, CML_valid
from models.OGM.OGM_main import OGM_main
from models.ACMo.ACMo_main import ACMo_main
from train_model.GBlending_train import GBleding_main
from utils.utils import get_logger



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
    parser.add_argument('--alpha', default=1.0, type=float, help='alpha in OGM-GE')

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
    parser.add_argument('--gpu_ids', default='1',
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
    return parser.parse_args()


class one_hot_CrossEntropy(torch.nn.Module):

    def __init__(self):
        super(one_hot_CrossEntropy, self).__init__()

    def forward(self, x, y):
        P_i = torch.nn.functional.softmax(x, dim=1)
        loss = y*torch.log(P_i + 0.0000001)
        loss = -torch.mean(torch.sum(loss, dim=1), dim=0)
        return loss


def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.
    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)
    Returns:
      stats: list of statistic of each class.
    """
    classes_num = target.shape[-1]
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc,
                # note acc is not class-wise, this is just to keep consistent with other metrics
                'acc': acc
                }
        stats.append(dict)

    return stats




def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    n_classes = 1
    #criterion = one_hot_CrossEntropy()
    # criterion = nn.BCEWithLogitsLoss()

    model.train()
    print("Start training ... ")

    _loss = 0

    loss_value_mm=[]
    loss_value_a=[]
    loss_value_v=[]

    cos_audio=[]
    cos_visual=[]

    num = [0.0 for _ in range(n_classes)]
    acc = [0.0 for _ in range(n_classes)]
    acc_a = [0.0 for _ in range(n_classes)]
    acc_v = [0.0 for _ in range(n_classes)]

    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param.data)
    #     print("requires_grad:", param.requires_grad)
    #     print("-----------------------------------")

    record_names_audio = []
    record_names_visual = []
    for name, param in model.named_parameters():
        if 'head' in name: 
            continue
        if ('audio' in name):
            record_names_audio.append((name, param))
            continue
        if ('visual' in name):
            record_names_visual.append((name, param))
            continue


    for step, (images,spec,  label) in tqdm(enumerate(dataloader)):

        # if(step>5):
        #     break
        optimizer.zero_grad()
        images = images.to(device)
        spec = spec.to(device)
        print(spec.shape)
        print(images.shape)
        label = label.to(device)
        out,out_a,out_v = model(spec.float(), images.float())

        loss_mm = criterion(out, label)
        loss_a=criterion(out_a,label)
        loss_v=criterion(out_v,label)

        loss=loss_mm
        loss.backward()

        optimizer.step()
        #scheduler.step()
        _loss += loss.item()

        # acc
        prediction = softmax(out)
        pred_a = softmax(out_a)
        pred_v = softmax(out_v)
        for j in range(images.shape[0]):
            ma = np.argmax(prediction[j].cpu().data.numpy())
            v = np.argmax(pred_v[j].cpu().data.numpy())
            a = np.argmax(pred_a[j].cpu().data.numpy())
            num[label[j]] += 1.0

            if np.asarray(label[j].cpu()) == ma:
                acc[label[j]] += 1.0
            if np.asarray(label[j].cpu()) == v:
                acc_v[label[j]] += 1.0
            if np.asarray(label[j].cpu()) == a:
                acc_a[label[j]] += 1.0

    accuracy = sum(acc) / sum(num)
    accuracy_a = sum(acc_a) / sum(num)
    accuracy_v = sum(acc_v) / sum(num)
    writer.add_scalars('Epoch Accuracy(train)', {'accuracy': accuracy,
                                          'audio_accuracy': accuracy_a,
                                          'visual accuracy': accuracy_v}, epoch)

    return _loss / len(dataloader),loss_value_mm,loss_value_a,loss_value_v,cos_audio,cos_visual


def valid(args, model, device, dataloader):
    n_classes = 31

    cri = nn.CrossEntropyLoss()
    _loss = 0

    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a= [0.0 for _ in range(n_classes)]
        acc_v= [0.0 for _ in range(n_classes)]

        for step, (images, spec,  label) in tqdm(enumerate(dataloader)):

            spec = spec.to(device)
            images = images.to(device)
            label = label.to(device)

            prediction_all = model(spec.unsqueeze(1).float(), images.float())
            # prediction = model(spec.float())
            # print(prediction[0])

            prediction = prediction_all[0]
            prediction_audio = prediction_all[1]
            prediction_visual = prediction_all[2]
            loss = cri(prediction, label)
            _loss += loss.item()

            for i, item in enumerate(label):

                ma = prediction[i].cpu().data.numpy()
                index_ma = np.argmax(ma)
                # print(index_ma, label_index)
                num[label[i]] += 1.0
                if index_ma == label[i]:
                    acc[label[i]] += 1.0
                
                ma_audio=prediction_audio[i].cpu().data.numpy()
                index_ma_audio = np.argmax(ma_audio)
                if index_ma_audio == label[i]:
                    acc_a[label[i]] += 1.0


                ma_visual=prediction_visual[i].cpu().data.numpy()
                index_ma_visual = np.argmax(ma_visual)
                if index_ma_visual == label[i]:
                    acc_v[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num),sum(acc_v) / sum(num), _loss / len(dataloader)


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
