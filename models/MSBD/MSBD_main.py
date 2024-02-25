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

def train_epoch(args, epoch, model, device, dataloader, optimizer):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    n_classes = 31
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
        

        loss = criterion(out, label) 
        loss_v = criterion(out_v, label)
        loss_a = criterion(out_a, label)

        prediction_a = softmax(out_a)
        prediction_v = softmax(out_v)
        if args.modulation_starts <= epoch <= args.modulation_ends:
            loss_RS = 1/out_a.shape[1] * torch.sum((out_a - out_v)**2, dim = 1)

            w = torch.tensor([0.0 for _ in range(len(out))])
            w = w.to(device)
            y_pred_a = prediction_a
            y_pred_a = y_pred_a.argmax(dim = -1)
            y_pred_v = prediction_v
            y_pred_v = y_pred_v.argmax(dim = -1)
            ps = torch.tensor([0.0 for _ in range(len(out))])
            ps = ps.to(device)
            pw = torch.tensor([0.0 for _ in range(len(out))])
            pw = pw.to(device)
            for i in range(len(out)):
                if y_pred_a[i] == label[i] or y_pred_v[i] == label[i]:
                    w[i] = max(prediction_a[i][label[i]], prediction_v[i][label[i]]) -  min(prediction_a[i][label[i]], prediction_v[i][label[i]])
                ps[i] = max(prediction_a[i][label[i]], prediction_v[i][label[i]])
                pw[i] = min(prediction_a[i][label[i]], prediction_v[i][label[i]])

            loss_KL = F.kl_div(ps, pw, reduction = 'none')
            w = w.reshape(1,-1)
            loss_KL = loss_KL.reshape(-1,1)
            loss_KL = torch.mm(w, loss_KL) / len(out)
            loss_RS = loss_RS.reshape(-1,1)
            loss_RS = torch.mm(w, loss_RS) / len(out)
            loss = loss + loss_v + loss_a + loss_RS.squeeze() + loss_KL.squeeze() ## erase the dim of 1

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

            loss = criterion(out, label) 
            
            valid_loss += loss
            acc += accuracy(out, label)*label.shape[0]
            acc_a += accuracy(out_a, label)*label.shape[0]
            acc_v += accuracy(out_v, label)*label.shape[0]
            count += label.shape[0]

    return acc/count, acc_a/count, acc_v/count, valid_loss/count


# def MSBD_main(args):
#     print(args)
#     setup_seed(args.random_seed)
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
#     gpu_ids = list(range(torch.cuda.device_count()))
#     device = torch.device('cuda:0')
#     model = AVClassifier(args)
#     model.to(device)
#     # model = torch.nn.DataParallel(model, device_ids=gpu_ids)

#     model.cuda()
#     # todo
#     # if args.audio_pretrain != 'None':
#     #     loaded_dict_audio = torch.load(args.audio_pretrain)
#     #     state_dict_audio = loaded_dict_audio
#     #     model.module.audio_net.load_state_dict(state_dict_audio, strict=False)
#     # if args.visual_pretrain != 'None':
#     #     loaded_dict_visual = torch.load(args.visual_pretrain)
#     #     state_dict_visual = loaded_dict_visual
#     #     model.module.visual_net.load_state_dict(state_dict_visual, strict=False)

#     if args.optimizer == 'sgd':
#         optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
#     elif args.optimizer == 'adam':
#         optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
#                                weight_decay=1e-4, amsgrad=False)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

#     train_dataset = AV_KS_Dataset(mode='train')
#     test_dataset = AV_KS_Dataset(mode='test')
#     val_dataset = AV_KS_Dataset(mode='val')

#     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
#                                   shuffle=True, num_workers=16, pin_memory=True)
#     test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
#                                  shuffle=False, num_workers=16)
#     val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
#                                  shuffle=False, num_workers=16)


#     if args.train:

#         best_acc = -1
#         writer_path = os.path.join(args.tensorboard_path)
#         if not os.path.exists(writer_path):
#             os.mkdir(writer_path)
#         log_name = 'model_{}_{}_{}_epoch{}_batch{}_lr{}_alpha{}'.format(
#             args.method, args.optimizer, args.dataset, args.epochs, args.batch_size, args.learning_rate, args.scaling)
#         writer = SummaryWriter(os.path.join(writer_path, log_name))  # alpha means scaling factor

#         ts = time.strftime('%Y_%m_%d %H:%M:%S', time.localtime())
#         print(ts)
#         save_dir = os.path.join(args.ckpt_path, f"{ts}_{args.method}")
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         logger = get_logger("train_logger", logger_dir=save_dir)

#         for epoch in range(args.epochs):
#             logger.info('Epoch: {}: '.format(epoch))
#             batch_loss, batch_loss_a, batch_loss_v, acc, acc_a, acc_v = train_epoch(args, epoch, model, device,
#                                                                  train_dataloader, optimizer, writer)
#             writer.add_scalars('Train', {'Total Accuracy': acc,
#                                               'Audio Accuracy': acc_a,
#                                               'Visual Accuracy': acc_v}, epoch)
#             scheduler.step()
#             acc, acc_a, acc_v, valid_loss = valid(args, model, device, val_dataloader)

#             writer.add_scalars('Loss', {'Total Loss': batch_loss,
#                                         'Audio Loss': batch_loss_a,
#                                         'Visual Loss': batch_loss_v}, epoch)

#             writer.add_scalars('Evaluation', {'Total Accuracy': acc,
#                                               'Audio Accuracy': acc_a,
#                                               'Visual Accuracy': acc_v}, epoch)

#             if acc > best_acc:
#                 best_acc = float(acc)

#                 if not os.path.exists(args.ckpt_path):
#                     os.mkdir(args.ckpt_path)

#                 model_name = 'best_model_{}_of_{}_epoch{}_batch{}_lr{}_alpha{}.pth'.format(
#                     args.method, args.optimizer, args.epochs, args.batch_size, args.learning_rate,
#                     args.alpha)

#                 saved_dict = {'saved_epoch': epoch,
#                               'fusion': args.fusion_method,
#                               'acc': acc,
#                               'model': model.state_dict(),
#                               'optimizer': optimizer.state_dict(),
#                               'scheduler': scheduler.state_dict()}

#                 save_dir = os.path.join(args.ckpt_path, model_name)

#                 torch.save(saved_dict, save_dir)
#                 logger.info('The best model has been saved at {}.'.format(save_dir))
#                 logger.info("Loss: {:.4f}, Acc: {:.4f}, Acc_a:{:.4f}, Acc_v:{:.4f}".format(batch_loss, acc, acc_a, acc_v))
#             else:
#                 logger.info("Loss: {:.4f}, Acc: {:.4f}, Best Acc: {:.4f}, Acc_a:{:.4f}, Acc_v:{:.4f}".format(batch_loss, acc,
#                                                                                                         best_acc, acc_a,
#                                                                                                         acc_v))
#     else:
#         # first load trained model
#         loaded_dict = torch.load(args.ckpt_path)
#         # epoch = loaded_dict['saved_epoch']
#         fusion = loaded_dict['fusion']
#         state_dict = loaded_dict['model']
#         # optimizer_dict = loaded_dict['optimizer']
#         # scheduler = loaded_dict['scheduler']

#         assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

#         model = model.load_state_dict(state_dict)
#         print('Trained model loaded!')

#         acc, acc_a, acc_v = valid(args, model, device, test_dataloader, epoch=1001)
#         print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))
