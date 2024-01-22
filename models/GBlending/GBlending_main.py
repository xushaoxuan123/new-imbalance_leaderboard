import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.av_dataset import AV_KS_Dataset
from models.models import AVClassifier_gb, RFClassifier_gb
from utils.utils import setup_seed

# def get_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', required=True, type=str,
#                         help='KineticSound, UCF-101')
#     parser.add_argument('--model', default='resnet18', type=str, choices=['resnet18'])
#     parser.add_argument('--modulation', default='origin', type=str, choices=['ours', 'origin'])

#     parser.add_argument('--batch_size', default=64, type=int)
#     parser.add_argument('--epochs', default=80, type=int)
#     parser.add_argument('--encoder_lr_decay', default=1.0, type=float, help='decay coefficient')
#     parser.add_argument('--super_epoch', default=5, type=int)

#     parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
#     parser.add_argument('--learning_rate', default=0.01, type=float, help='initial learning rate')
#     parser.add_argument('--lr_decay_step', default=40, type=int, help='where learning rate decays')
#     parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

#     parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
#     parser.add_argument('--modulation_ends', default=60, type=int, help='where modulation ends')
#     parser.add_argument('--alpha', default=4.0, type=float, help='alpha in OGM-GE')

#     parser.add_argument('--train', action='store_true', help='turn on train mode')
#     parser.add_argument('--log_path', default='/home/ruoxuan_feng/cooperative/log_gb', type=str, help='path to save tensorboard logs')

#     parser.add_argument('--random_seed', default=0, type=int)
#     parser.add_argument('--gpu_ids', default='2, 3', type=str, help='GPU ids')

#     return parser.parse_args()

def super_epoch_origin(args, model, temp_model, device, train_dataloader, test_dataloader, optimizer):
  pre_a_loss_train = 0.0
  pre_v_loss_train = 0.0
  pre_av_loss_train = 0.0
  now_a_loss_train = 0.0
  now_v_loss_train = 0.0
  now_av_loss_train = 0.0
  pre_a_loss_test = 0.0
  pre_v_loss_test = 0.0
  pre_av_loss_test = 0.0
  now_a_loss_test = 0.0
  now_v_loss_test = 0.0
  now_av_loss_test = 0.0
  _loss_av = 0.0
  _loss_a = 0.0
  _loss_v = 0.0
  _loss_av_test = 0.0
  _loss_a_test = 0.0
  _loss_v_test = 0.0
  criterion = nn.CrossEntropyLoss()
  softmax = nn.Softmax(dim=1)
  print('super begin')
  #audio flow
  temp_model.load_state_dict(model.state_dict(),strict=True)
  for epoch in range(args.super_epoch):
    _loss_a = 0.0
    _loss_a_test = 0.0
    temp_model.train()
    for step, (image, spec, label) in enumerate(train_dataloader):
      optimizer.zero_grad()
      # image = image.to(device)
      spec = spec.to(device)
      # print(spec.shape, image.shape)
      label = label.to(device)
      out_a = temp_model(spec.unsqueeze(1).float(), types = 1)
      loss_a = criterion(out_a, label)
      loss_a.backward()
      _loss_a += loss_a.item()
      optimizer.step()

    _loss_a /= len(train_dataloader)

    if epoch == 0 or epoch == args.super_epoch - 1:
      with torch.no_grad():
        temp_model.eval()
        for step, (image, spec, label) in enumerate(test_dataloader):
          spec = spec.to(device)
          label = label.to(device)
          out_a = temp_model(spec.unsqueeze(1).float(), types = 1)
          loss_a = criterion(out_a, label)
          _loss_a_test += loss_a.item()

        _loss_a_test /= len(test_dataloader)

        if epoch == 0:
          pre_a_loss_train = _loss_a
          pre_a_loss_test = _loss_a_test
        else:
          now_a_loss_train = _loss_a
          now_a_loss_test = _loss_a_test

  # visual
  temp_model.load_state_dict(model.state_dict(),strict=True)
  for epoch in range(args.super_epoch):
    _loss_v = 0.0
    _loss_v_test = 0.0
    temp_model.train()
    for step, (image, spec, label) in enumerate(train_dataloader):
      optimizer.zero_grad()
      # image = image.to(device)
      image = image.to(device)
      label = label.to(device)
      out_v = temp_model(visual = image.float(), types = 2)
      loss_v = criterion(out_v, label)
      loss_v.backward()
      _loss_v += loss_v.item()
      optimizer.step()

    _loss_v /= len(train_dataloader)

    if epoch == 0 or epoch == args.super_epoch - 1:
      with torch.no_grad():
        temp_model.eval()
        for step, (image, spec, label,) in enumerate(test_dataloader):
          image = image.to(device)
          label = label.to(device)
          out_v = temp_model(visual = image.float(), types = 2)
          loss_v = criterion(out_v, label)
          _loss_v_test += loss_v.item()

        _loss_v_test /= len(test_dataloader)

        if epoch == 0:
          pre_v_loss_train = _loss_v
          pre_v_loss_test = _loss_v_test
        else:
          now_v_loss_train = _loss_v
          now_v_loss_test = _loss_v_test

  # all
  temp_model.load_state_dict(model.state_dict(),strict=True)
  for epoch in range(args.super_epoch):
    temp_model.train()
    _loss_av = 0.0
    _loss_av_test = 0.0
    for step, (image, spec, label) in enumerate(train_dataloader):
      optimizer.zero_grad()
      # image = image.to(device)
      spec = spec.to(device)
      image = image.to(device)
      label = label.to(device)
      _a, _v, out_av = temp_model(spec.unsqueeze(1).float(),image.float())
      loss_av = criterion(out_av, label)
      loss_av.backward()
      _loss_av += loss_av.item()
      optimizer.step()

    _loss_av /= len(train_dataloader)

    if epoch == 0 or epoch == args.super_epoch - 1:
      with torch.no_grad():
        temp_model.eval()
        for step, (image, spec, label,) in enumerate(test_dataloader):
          spec = spec.to(device)
          image = image.to(device)
          label = label.to(device)
          _a, _v, out_av = temp_model(spec.unsqueeze(1).float(),image.float())
          loss_av = criterion(out_av, label)
          _loss_av_test += loss_av.item()

        _loss_av_test /= len(test_dataloader)

        if epoch == 0:
          pre_av_loss_train = _loss_av
          pre_av_loss_test = _loss_av_test
        else:
          now_av_loss_train = _loss_av
          now_av_loss_test = _loss_av_test
      
  g_a = pre_a_loss_test - now_a_loss_test
  o_a_pre = pre_a_loss_test - pre_a_loss_train
  o_a_now = now_a_loss_test - now_a_loss_train
  o_a = o_a_now - o_a_pre
  weight_a = abs(g_a / (o_a * o_a))

  g_v = pre_v_loss_test - now_v_loss_test
  o_v_pre = pre_v_loss_test - pre_v_loss_train
  o_v_now = now_v_loss_test - now_v_loss_train
  o_v = o_v_now - o_v_pre
  weight_v = abs(g_v / (o_v * o_v))

  g_av = pre_av_loss_test - now_av_loss_test
  o_av_pre = pre_av_loss_test - pre_av_loss_train
  o_av_now = now_av_loss_test - now_av_loss_train
  o_av = o_av_now - o_av_pre
  weight_av = abs(g_av / (o_av * o_av))

  sums = weight_a + weight_v + weight_av

  weight_a /= sums
  weight_v /= sums
  weight_av /= sums

  print(pre_a_loss_train, pre_v_loss_train, pre_av_loss_train)
  print(now_a_loss_train, now_v_loss_train, now_av_loss_train)
  return weight_a, weight_v, weight_av


def train_epoch(args, epoch, model, device, dataloader, optimizer, weight_a, weight_v, weight_av, writer=None):
  criterion = nn.CrossEntropyLoss()
  softmax = nn.Softmax(dim=1)
  relu = nn.ReLU(inplace=True)
  tanh = nn.Tanh()

  model.train()
  print("Start training ... ")
  print(weight_a, weight_v, weight_av)

  _loss = 0
  _loss_a = 0
  _loss_v = 0
  for step, (image, spec, label, index) in enumerate(dataloader):
    B = image.size()[0]
    optimizer.zero_grad()

    image = image.to(device)
    spec = spec.to(device)
    label = label.to(device)
    out_a, out_v, out = model(spec.unsqueeze(1).float(), image.float())

    # out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) +
    #                  model.module.fusion_module.fc_out.bias / 2)
    # out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
    #                  model.module.fusion_module.fc_out.bias / 2)
    loss_a = criterion(out_a, label)
    loss_v = criterion(out_v, label)
    loss_av = criterion(out, label)

    if args.modulation_starts <= epoch < args.modulation_ends:
      loss = weight_a * loss_a + weight_v * loss_v + weight_av * loss_av
    
    else:
      loss = loss_av

    loss.backward()

    # prediction = softmax(out)
    # pred_v = softmax(out_v)
    # pred_a = softmax(out_a)

    optimizer.step()
    _loss += loss.item()
    _loss_a += loss_a.item()
    _loss_v += loss_v.item()
  return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)

def valid(args, model, device, dataloader):
  softmax = nn.Softmax(dim=1)

  if args.dataset == 'KineticSound':
      n_classes = 31
  elif args.dataset == 'UCF-101':
      n_classes = 101
  cri = nn.CrossEntropyLoss()
  _loss = 0

  with torch.no_grad():
      model.eval()
      num = [0.0 for _ in range(n_classes)]
      acc = [0.0 for _ in range(n_classes)]
      acc_a = [0.0 for _ in range(n_classes)]
      acc_v = [0.0 for _ in range(n_classes)]

      for step, (image, spec, label) in enumerate(dataloader):
        image = image.to(device)
        spec = spec.to(device)
        label = label.to(device)

        out_a, out_v, out = model(spec.unsqueeze(1).float(), image.float())

        # out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) +
        #              model.module.fusion_module.fc_out.bias / 2)
        # out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
        #              model.module.fusion_module.fc_out.bias / 2)

        prediction = softmax(out)
        pred_v = softmax(out_v)
        pred_a = softmax(out_a)

        loss = cri(out, label)
        _loss+=loss.item()

        for i, item in enumerate(label):
          ma = prediction[i].cpu().data.numpy()
          index_ma = np.argmax(ma)
          v = pred_v[i].cpu().data.numpy()
          index_v = np.argmax(v)
          a = pred_a[i].cpu().data.numpy()
          index_a = np.argmax(a)
          num[label[i]] += 1.0
          if index_ma == label[i]:
              acc[label[i]] += 1.0
          if index_v == label[i]:
              acc_v[label[i]] += 1.0
          if index_a == label[i]:
              acc_a[label[i]] += 1.0
  
  return sum(acc) / sum(num),  sum(acc_a) / sum(num) , sum(acc_v) / sum(num), _loss / len(dataloader)

# def GBleding_main(args):
#   cona_all = []
#   conv_all = []
#   # print(args)

#   setup_seed(args.random_seed)
#   os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
#   gpu_ids = list(range(torch.cuda.device_count()))

#   device = torch.device('cuda:0')

#   if args.dataset == 'KineticSound':
#     model = AVClassifier_gb(args)
#     temp_model = AVClassifier_gb(args)
#   elif args.dataset == 'UCF-101':
#     model = RFClassifier_gb(args)
#     temp_model = RFClassifier_gb(args)
#   model.to(device)
#   model = torch.nn.DataParallel(model, device_ids=gpu_ids)
#   model.cuda()

#   temp_model.to(device)
#   temp_model = torch.nn.DataParallel(temp_model, device_ids=gpu_ids)
#   temp_model.cuda()

#   if args.dataset == 'KineticSound':
#     train_dataset = AV_KS_Dataset(mode='train')
#     test_dataset = AV_KS_Dataset(mode='test')
  
  
#   train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
#                                   shuffle=True, num_workers=16,pin_memory=True)
#   test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
#                                  shuffle=False, num_workers=16)

#   # params = [
#   #   {"params": model.module.audio_net.parameters(), "lr": args.learning_rate * args.encoder_lr_decay},
#   #   {"params": model.module.visual_net.parameters(), "lr": args.learning_rate * args.encoder_lr_decay},
#   #   {"params": model.module.fusion_module.parameters(), "lr": args.learning_rate},
#   # ]

#   if args.optimizer == 'sgd':
#         optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
#         temp_optimizer = optim.SGD(temp_model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
#   elif args.optimizer == 'adam':
#         optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
#         temp_optimizer = optim.Adam(temp_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)

#   scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)
#   temp_scheduler = optim.lr_scheduler.StepLR(temp_optimizer, args.lr_decay_step, args.lr_decay_ratio)

#   if args.train:
#     best_acc = 0.0

#     for epoch in range(args.epochs):

#       print('Epoch: {}: '.format(epoch))
#       writer_path = os.path.join(args.tensorboard_path)
#       if not os.path.exists(writer_path):
#         os.mkdir(writer_path)
#       log_name = '{}_{}_{}_{}_epochs{}_batch{}_lr{}_alpha{}'.format(args.optimizer,  args.dataset, args.fusion_method, args.model, args.epochs, args.batch_size, args.learning_rate, args.alpha)
#       writer = SummaryWriter(os.path.join(writer_path, log_name))
#       if epoch % args.super_epoch == 0:
#         wa, wv, wav = super_epoch_origin(args, model, temp_model, device, train_dataloader, test_dataloader, temp_optimizer)

#       batch_loss = train_epoch(args, epoch, model, device, train_dataloader, optimizer, wa, wv, wav, writer)
#       scheduler.step()
#       temp_scheduler.step()
#       acc, acc_a, acc_v, val_loss = valid(args, model, device, test_dataloader)

#       writer.add_scalars('Loss', {'Total Loss': batch_loss, 'Val Loss': val_loss}, epoch)

#       writer.add_scalars('Evaluation', {'Total Accuracy': acc,
#                                               'Audio Accuracy': acc_a,
#                                               'Visual Accuracy': acc_v}, epoch)

#       if acc > best_acc:
#         best_acc = float(acc)

#         model_name = 'best_model_of_{}_{}_{}_{}_batch{}_lr{}_alpha{}.pth'.format(args.optimizer,  args.dataset, args.modulation, args.model, args.batch_size, args.learning_rate, args.alpha)
            
#         saved_dict = {'saved_epoch': epoch,
#                       'acc': acc,
#                       'model': model.state_dict(),
#                       'optimizer': optimizer.state_dict(),
#                       'scheduler': scheduler.state_dict()}

#         save_dir = os.path.join(args.log_path, model_name)

#         torch.save(saved_dict, save_dir)
#         print('The best model has been saved at {}.'.format(save_dir))
#         print("Loss: {:.4f}, Acc: {:.4f}, Acc_a:{:.4f}, Acc_v:{:.4f}, Val Loss: {:.4f}".format(batch_loss, acc,acc_a,acc_v, val_loss))

#       else:
#         print("Loss: {:.4f}, Acc: {:.4f}, Best Acc: {:.4f}, Acc_a:{:.4f}, Acc_v:{:.4f}, Val Loss: {:.4f}".format(batch_loss, acc, best_acc,acc_a,acc_v, val_loss))
