import argparse
import time
from operator import mod
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from dataset.av_dataset import AV_KS_Dataset
from models.models import AVClassifier
from utils.utils import setup_seed, weight_init, re_init,get_logger, accuracy
from models.Loss import NCELoss
EPISILON = 1e-10

# todo add logger
# NCE loss is used in supplementary material 蒸馏损失

def train_epoch(args, epoch, model, device, dataloader, optimizer, writer, logger=None):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    n_classes = 31
    NCE = NCELoss(args.temperature, args.EPISILON)
    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0
    num = [0.0 for _ in range(n_classes)]
    acc = [0.0 for _ in range(n_classes)]
    acc_a = [0.0 for _ in range(n_classes)]
    acc_v = [0.0 for _ in range(n_classes)]

    for step, (image, spec, label) in enumerate(dataloader):
        optimizer.zero_grad()

        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)
        out, a, v = model(spec.unsqueeze(1).float(), image.float())
        nce_loss = NCE(a, v, label)
        # print(a.shape, v.shape, model.head.weight.shape)

        ## our modality-wise normalization on weight and feature
        if args.method == 'MMCosine':
            out_a = torch.mm(F.normalize(a, dim=1),
                             F.normalize(torch.transpose(model.head.weight[:, :512], 0, 1),
                                         dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
            out_v = torch.mm(F.normalize(v, dim=1),
                             F.normalize(torch.transpose(model.head.weight[:, 512:], 0, 1),
                                         dim=0))
            out_a = out_a * args.scaling
            out_v = out_v * args.scaling
            out = out_a + out_v
        else:
            out_a = (torch.mm(a, torch.transpose(model.head.weight[:, :512], 0, 1)) +
                     model.head.bias / 2)
            out_v = (torch.mm(v, torch.transpose(model.head.weight[:, 512:], 0, 1)) +
                     model.head.bias / 2)

        if args.use_tensorboard:
            label_onehot = nn.functional.one_hot(label, num_classes=out_v.size(1))
            fy_v = torch.mean(torch.sum(out_v * label_onehot, dim=1))
            fy_a = torch.mean(torch.sum(out_a * label_onehot, dim=1))
            iteration = epoch * len(dataloader) + step

            writer.add_scalar('data/logit_a', fy_a, iteration)
            writer.add_scalar('data/logit_v', fy_v, iteration)

        # acc
        for i in range(n_classes):
            ma = out[i].cpu().data.numpy()
            index_ma = np.argmax(ma)
            v = out_v[i].cpu().data.numpy()
            index_v = np.argmax(v)
            a = out_a[i].cpu().data.numpy()
            index_a = np.argmax(a)
            num[label[i]] += 1.0
            if index_ma == label[i]:
                acc[label[i]] += 1.0
            if index_v == label[i]:
                acc_v[label[i]] += 1.0
            if index_a == label[i]:
                acc_a[label[i]] += 1.0

        loss = criterion(out, label) + args.lam * nce_loss
        loss_v = criterion(out_v, label)
        loss_a = criterion(out_a, label)
        loss.backward()

        optimizer.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)
    n_classes = 31 # KS_dataset
    count = 0
    acc = 0.
    acc_v = 0.
    acc_a = 0.

    criterion = nn.CrossEntropyLoss()
    loss = 0
    valid_loss = 0
    NCE = NCELoss(args.temperature, args.EPISILON)
    with torch.no_grad():
        model.eval()
        # TODO: more flexible

        for step, (image, spec, label) in enumerate(dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)
            out, a, v = model(spec.unsqueeze(1).float(), image.float())

            # approximate uni-modal evaluation
            if args.method == 'MMCosine':
                out_a = torch.mm(F.normalize(a, dim=1),
                                 F.normalize(torch.transpose(model.head.weight[:, :512], 0, 1),
                                             dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
                out_v = torch.mm(F.normalize(v, dim=1),
                                 F.normalize(torch.transpose(model.head.weight[:, 512:], 0, 1),
                                             dim=0))
                out_a = out_a * args.scaling
                out_v = out_v * args.scaling
                out = out_a + out_v
            else:
                out_a = (torch.mm(a, torch.transpose(model.head.weight[:, :512], 0, 1)) +
                         model.head.bias / 2)
                out_v = (torch.mm(v, torch.transpose(model.head.weight[:, 512:], 0, 1)) +
                         model.head.bias / 2)
            nce_loss = NCE(a, v, label)
            loss = criterion(out, label) + args.lam * nce_loss
            
            valid_loss += loss
            acc += accuracy(out, label)*label.shape[0]
            acc_a += accuracy(out_a, label)*label.shape[0]
            acc_v += accuracy(out_v, label)*label.shape[0]
            count += label.shape[0]

    return acc/count, acc_a/count, acc_v/count, valid_loss/count


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
        writer_path = os.path.join(args.tensorboard_path)
        if not os.path.exists(writer_path):
            os.mkdir(writer_path)
        log_name = 'model_{}_{}_{}_epoch{}_batch{}_lr{}_alpha{}'.format(
            args.method, args.optimizer, args.dataset, args.epochs, args.batch_size, args.learning_rate, args.scaling)
        writer = SummaryWriter(os.path.join(writer_path, log_name))  # alpha means scaling factor

        ts = time.strftime('%Y_%m_%d %H:%M:%S', time.localtime())
        print(ts)
        save_dir = os.path.join(args.ckpt_path, f"{ts}_{args.method}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logger = get_logger("train_logger", logger_dir=save_dir)

        for epoch in range(args.epochs):
            logger.info('Epoch: {}: '.format(epoch))
            batch_loss, batch_loss_a, batch_loss_v, acc, acc_a, acc_v = train_epoch(args, epoch, model, device,
                                                                 train_dataloader, optimizer, writer)
            writer.add_scalars('Train', {'Total Accuracy': acc,
                                              'Audio Accuracy': acc_a,
                                              'Visual Accuracy': acc_v}, epoch)
            print('train_loss:',acc)
            scheduler.step()
            acc, acc_a, acc_v, valid_loss = valid(args, model, device, val_dataloader)

            writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                        'Audio Loss': batch_loss_a,
                                        'Visual Loss': batch_loss_v}, epoch)

            writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                              'Audio Accuracy': acc_a,
                                              'Visual Accuracy': acc_v}, epoch)

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'best_model_{}_of_{}_epoch{}_batch{}_lr{}_alpha{}.pth'.format(
                    args.method, args.optimizer, args.epochs, args.batch_size, args.learning_rate,
                    args.alpha)

                saved_dict = {'saved_epoch': epoch,
                              'fusion': args.fusion_method,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                # torch.save(saved_dict, save_dir)
                # logger.info('The best model has been saved at {}.'.format(save_dir))
                logger.info("Loss: {:.4f}, Acc: {:.4f}, Acc_a:{:.4f}, Acc_v:{:.4f}".format(batch_loss, acc, acc_a, acc_v))
            else:
                logger.info("Loss: {:.4f}, Acc: {:.4f}, Best Acc: {:.4f}, Acc_a:{:.4f}, Acc_v:{:.4f}".format(batch_loss, acc,
                                                                                                        best_acc, acc_a,
                                                                                                        acc_v))
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


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, defult="CREMAD", type=str,
                        help='CREMAD')
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['concat', 'gated', 'film'])
    parser.add_argument('--mmcosine', default=False, type=bool, help='whether to involve mmcosine')
    parser.add_argument('--scaling', default=10, type=float, help='scaling parameter in mmCosine')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=100, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    # parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--ckpt_path', default='./log', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=True, type=bool, help='whether to visualize')
    # parser.add_argument('--tensorboard_path', required=True, type=str, help='path to save tensorboard logs')
    parser.add_argument('--tensorboard_path', default='/home/ruize_xu/CD/log_normal', type=str,
                        help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0, 1', type=str, help='GPU ids')
    parser.add_argument("--lam", type=float, default=0)

    ## uni-modal pretrained checkpoint(mainly for ssw)
    parser.add_argument('--audio_pretrain', default='None', type=str, help='path of pretrained audio resnet')
    parser.add_argument('--visual_pretrain', default='None', type=str, help='path of pretrained visual resnet')

    return parser.parse_args()