import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.av_dataset import AV_KS_Dataset
# from dataset.CGMNIST import CGMNISTDataset
# from dataset.CramedDataset import CramedDataset
# from dataset.AVEDataset import AVEDataset
# from dataset.dataset import AVDataset
from models.models import AVClassifier # , CGClassifier
from utils.utils import setup_seed, weight_init, get_logger
# from dataset.VGGSoundDataset import VGGSound
import time


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='OGM_GE', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE', 'Acc', 'Proto'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--fps', default=1, type=int, help='Extract how many frames in a second')
    parser.add_argument('--num_frame', default=1, type=int, help='use how many frames for train')

    parser.add_argument('--optimizer', default='SGD', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--momentum_coef', default=0.2, type=float)
    parser.add_argument('--proto_update_freq', default=50, type=int, help='steps')

    # parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=100, type=int, help='where modulation ends')
    parser.add_argument('--alpha', default=1.0, type=float, help='alpha in Proto')

    parser.add_argument('--ckpt_path', default='ckpt', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', action='store_true', help='whether to visualize')
    parser.add_argument('--tensorboard_path', default='logs', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--gpu', type=int, default=0)  # gpu
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    # args = parser.parse_args()
    #
    # args.use_cuda = torch.cuda.is_available() and not args.no_cuda

    return parser.parse_args()


def EU_dist(x1, x2):
    d_matrix = torch.zeros(x1.shape[0], x2.shape[0]).to(x1.device)
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            d = torch.sqrt(torch.dot((x1[i] - x2[j]), (x1[i] - x2[j])))
            d_matrix[i, j] = d
    return d_matrix


def dot_product_angle_tensor(v1, v2):
    vector_dot_product = torch.dot(v1, v2)
    arccos = torch.acos(vector_dot_product / (torch.norm(v1, p=2) * torch.norm(v2, p=2)))
    angle = np.degrees(arccos.data.cpu().numpy())
    return arccos, angle


def grad_amplitude_diff(v1, v2):
    len_v1 = torch.norm(v1, p=2)
    len_v2 = torch.norm(v2, p=2)
    return len_v1, len_v2, len_v1 - len_v2


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler,
                audio_proto, visual_proto, writer, logger):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()

    total_batch = len(dataloader)
    n_classes = 31
    _loss = 0.
    _loss_a = 0.
    _loss_v = 0.
    _loss_p_a = 0.
    _loss_p_v = 0.

    _a_angle = 0.
    _v_angle = 0.
    _a_diff = 0.
    _v_diff = 0.
    _ratio_a = 0.
    _ratio_a_p = 0.

    num = [0.0 for _ in range(n_classes)]
    acc = [0.0 for _ in range(n_classes)]
    acc_a = [0.0 for _ in range(n_classes)]
    acc_v = [0.0 for _ in range(n_classes)]
    for step, (image, spec, label) in enumerate(dataloader):
        spec = spec.to(device)  # B x 257 x 1004(CREMAD 299)
        image = image.to(device)  # B x 1(image count) x 3 x 224 x 224
        label = label.to(device)  # B

        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        if args.dataset != 'CGMNIST':
            a, v, out = model(spec.unsqueeze(1).float(), image.float())
        else:
            a, v, out = model(spec, image)  # gray colored
        if args.fusion_method == 'sum':
            out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
                     model.fusion_module.fc_y.bias)
            out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
                     model.fusion_module.fc_x.bias)
        elif args.fusion_method == 'concat':
            weight_size = model.head.weight.size(1)
            out_v = (torch.mm(v, torch.transpose(model.head.weight[:, weight_size // 2:], 0, 1))
                     + model.head.bias / 2)
            out_a = (torch.mm(a, torch.transpose(model.head.weight[:, :weight_size // 2], 0, 1))
                     + model.head.bias / 2)
        elif args.fusion_method == 'film':
            out_v = out
            out_a = out
        elif args.fusion_method == 'gated':
            out_v = out
            out_a = out

        audio_sim = -EU_dist(a, audio_proto)  # B x n_class
        visual_sim = -EU_dist(v, visual_proto)  # B x n_class
        # print('sim: ', audio_sim[0][0].data, visual_sim[0][0].data, a[0][0].data, v[0][0].data)

        if args.method == 'PMR' and args.modulation_starts <= epoch <= args.modulation_ends:
            score_a_p = sum([softmax(audio_sim)[i][label[i]] for i in range(audio_sim.size(0))])
            score_v_p = sum([softmax(visual_sim)[i][label[i]] for i in range(visual_sim.size(0))])
            ratio_a_p = score_a_p / score_v_p

            score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
            score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])
            ratio_a = score_a / score_v

            loss_proto_a = criterion(audio_sim, label)
            loss_proto_v = criterion(visual_sim, label)

            if ratio_a_p > 1:
                beta = 0  # audio coef
                lam = 1 * args.alpha  # visual coef
            elif ratio_a_p < 1:
                beta = 1 * args.alpha
                lam = 0
            else:
                beta = 0
                lam = 0
            loss = criterion(out, label) + beta * loss_proto_a + lam * loss_proto_v
            loss_v = criterion(out_v, label)
            loss_a = criterion(out_a, label)
        else:
            loss = criterion(out, label)
            loss_proto_v = criterion(visual_sim, label)
            loss_proto_a = criterion(audio_sim, label)
            loss_v = criterion(out_v, label)
            loss_a = criterion(out_a, label)

            score_a_p = sum([softmax(audio_sim)[i][label[i]] for i in range(audio_sim.size(0))])
            score_v_p = sum([softmax(visual_sim)[i][label[i]] for i in range(visual_sim.size(0))])
            ratio_a_p = score_a_p / score_v_p
            score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
            score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])
            ratio_a = score_a / score_v

        writer.add_scalar('loss/step', loss, (epoch - 1) * total_batch + step)

        prediction = softmax(out)
        pred_a = softmax(out_a)
        pred_v = softmax(out_v)
        audio_sim = -EU_dist(a, audio_proto)  # B x n_class
        visual_sim = -EU_dist(v, visual_proto)  # B x n_class
        # todo more acc to print
        pred_v_p = softmax(visual_sim)
        pred_a_p = softmax(audio_sim)
        for j in range(image.shape[0]):
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

        if args.fusion_method == 'sum' or args.fusion_method == 'concat':
            loss.backward()
        else:
            loss.backward()
            logger.info('ratio: ', ratio_a, ratio_a_p)
            a_angle = 0
            v_angle = 0
            _a_angle += a_angle
            _v_angle += v_angle
        # logger.info('loss: ', loss.data, 'loss_p_v: ', loss_proto_v.data, 'loss_p_a: ', loss_proto_a.data,
        #       'loss_v: ', loss_v.data, 'loss_a: ', loss_a.data)
        if step % 100 == 0:
            logger.info(
                'EPOCH:[{:3d}/{:3d}]--STEP:[{:5d}/{:5d}]--{}--Loss:{:.4f}--lr:{}'.format(epoch, args.epochs, step,
                                                                                         total_batch, 'Train',
                                                                                         loss.item(),
                                                                                         [group['lr'] for group in
                                                                                          optimizer.param_groups]))

        optimizer.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()
        _loss_p_a += loss_proto_a.item()
        _loss_p_v += loss_proto_v.item()
        _ratio_a += ratio_a
        _ratio_a_p += ratio_a_p

    if args.optimizer == 'sgd':
        scheduler.step()
    # f_angle.close()
    accuracy = sum(acc) / sum(num)
    accuracy_a = sum(acc_a) / sum(num)
    accuracy_v = sum(acc_v) / sum(num)
    writer.add_scalars('Epoch Accuracy(train)', {'accuracy': accuracy,
                                                 'accuracy audio': accuracy_a,
                                                 'accuracy visual': accuracy_v}, epoch)
    logger.info(
        'EPOCH:[{:3d}/{:3d}]--{}--acc:{:.4f}--acc_a:{:.4f}--acc_v:{:.4f}-Alpha:{}'.format(epoch, args.epochs,
                                                                                          'Train', accuracy,
                                                                                          accuracy_a, accuracy_v,
                                                                                          args.alpha))

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), \
           _loss_p_a / len(dataloader), _loss_p_v / len(dataloader), \
           _a_angle / len(dataloader), _v_angle / len(dataloader), \
           _ratio_a / len(dataloader), _ratio_a_p / len(dataloader), _a_diff / len(dataloader), _v_diff / len(dataloader)


def valid(args, model, device, dataloader, audio_proto, visual_proto, epoch, writer, logger):
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    n_classes = 31
    total_batch = len(dataloader)
    start_time = time.time()
    _loss = 0.

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]
        acc_a_p = [0.0 for _ in range(n_classes)]
        acc_v_p = [0.0 for _ in range(n_classes)]

        for step, (image, spec, label) in enumerate(dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            if args.dataset != 'CGMNIST':
                a, v, out = model(spec.unsqueeze(1).float(), image.float())
            else:
                a, v, out = model(spec, image)  # gray colored

            if args.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
                         model.fusion_module.fc_y.bias)
                out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
                         model.fusion_module.fc_x.bias)
            elif args.fusion_method == 'concat':
                weight_size = model.head.weight.size(1)
                out_v = (torch.mm(v, torch.transpose(model.head.weight[:, weight_size // 2:], 0, 1))
                         + model.head.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.head.weight[:, :weight_size // 2], 0, 1))
                         + model.head.bias / 2)
            elif args.fusion_method == 'film':
                out_v = out
                out_a = out
            elif args.fusion_method == 'gated':
                out_v = out
                out_a = out

            loss = criterion(out, label)
            _loss += loss.item()
            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            audio_sim = -EU_dist(a, audio_proto)  # B x n_class
            visual_sim = -EU_dist(v, visual_proto)  # B x n_class
            # print(audio_sim, visual_sim, (audio_sim != audio_sim).any(), (visual_sim != visual_sim).any())
            pred_v_p = softmax(visual_sim)
            pred_a_p = softmax(audio_sim)
            # print('pred_p: ', (pred_a_p != pred_a_p).any(), (pred_v_p != pred_v_p).any())

            for i in range(image.shape[0]):
                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                v_p = np.argmax(pred_v_p[i].cpu().data.numpy())
                a_p = np.argmax(pred_a_p[i].cpu().data.numpy())
                num[label[i]] += 1.0

                # pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v_p:
                    acc_v_p[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a_p:
                    acc_a_p[label[i]] += 1.0

            if step % 20 == 0:
                logger.info('EPOCH:[{:03d}/{:03d}]--STEP:[{:05d}/{:05d}]--{}--loss:{}'.format(epoch, args.epochs, step,
                                                                                              total_batch, 'Validate',
                                                                                              loss))

    accuracy = sum(acc) / sum(num)
    accuracy_a = sum(acc_a) / sum(num)
    accuracy_v = sum(acc_v) / sum(num)
    writer.add_scalars('Accuracy(Test)', {'accuracy': accuracy,
                                          'audio_accuracy': accuracy_a,
                                          'visual accuracy': accuracy_v}, epoch)

    model.mode = 'train'
    end_time = time.time()
    elapse_time = end_time - start_time
    logger.info(
        'EPOCH:[{:03d}/{:03d}]--{}--Elapse time:{:.2f}--Accuracy:{:.4f}--acc_a:{:.4f}--acc_v:{:.4f}'.format(epoch,
                                                                                                            args.epochs,
                                                                                                            'Validate',
                                                                                                            elapse_time,
                                                                                                            accuracy,
                                                                                                            accuracy_a,
                                                                                                            accuracy_v))

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num), sum(acc_a_p) / sum(num), sum(acc_v_p) / sum(num), _loss/sum(num)


def calculate_prototype(args, model, dataloader, device, epoch, a_proto=None, v_proto=None):
    # todo customed output of prototype
    n_classes = 31

    audio_prototypes = torch.zeros(n_classes, args.embed_dim).to(device)
    visual_prototypes = torch.zeros(n_classes, args.embed_dim).to(device)
    count_class = [0 for _ in range(n_classes)]

    # calculate prototype
    model.eval()
    with torch.no_grad():
        sample_count = 0
        all_num = len(dataloader)
        for step, (image, spec, label) in enumerate(dataloader):
            spec = spec.to(device)  # B x 257 x 1004
            image = image.to(device)  # B x 3(image count) x 3 x 224 x 224
            label = label.to(device)  # B

            # TODO: make it simpler and easier to extend
            if args.dataset != 'CGMNIST':
                a, v, out = model(spec.unsqueeze(1).float(), image.float())
            else:
                a, v, out = model(spec, image)  # gray colored

            for c, l in enumerate(label):
                l = l.long()
                count_class[l] += 1
                audio_prototypes[l, :] += a[c, :]
                visual_prototypes[l, :] += v[c, :]
                # if l == 22:
                #     print('fea_a', a[c, :], audio_prototypes[l, :])

            sample_count += 1
            if args.dataset == 'AVE':
                pass
            else:
                if sample_count >= all_num // 10:
                    break
    for c in range(audio_prototypes.shape[0]):
        audio_prototypes[c, :] /= count_class[c]
        visual_prototypes[c, :] /= count_class[c]

    if epoch <= 0:
        audio_prototypes = audio_prototypes
        visual_prototypes = visual_prototypes
    else:
        audio_prototypes = (1 - args.momentum_coef) * audio_prototypes + args.momentum_coef * a_proto
        visual_prototypes = (1 - args.momentum_coef) * visual_prototypes + args.momentum_coef * v_proto
    return audio_prototypes, visual_prototypes


def PMR_main(args):
    args.momentum_coef = 0.5
    args.embed_dim = 512
    print(args)
    setup_seed(args.random_seed)
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

    ts = time.strftime('%Y_%m_%d %H:%M:%S', time.localtime())
    save_dir = os.path.join(args.ckpt_path, f"{ts}_PMR")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = get_logger("train_logger", logger_dir=save_dir)

    if args.train:
        # tensorboard
        writer_path = os.path.join(args.tensorboard_path)
        if not os.path.exists(writer_path):
            os.mkdir(writer_path)
        log_name = 'model_{}_{}_{}_epoch{}_batch{}_lr{}_alpha{}'.format(
            args.method, args.optimizer, args.dataset, args.epochs, args.batch_size, args.learning_rate, args.alpha)
        writer = SummaryWriter(os.path.join(writer_path, log_name))

        best_acc = 0
        epoch = 0
        audio_proto, visual_proto = calculate_prototype(args, model, train_dataloader, device, epoch)

        for epoch in range(1, args.epochs+1):
            logger.info('Epoch: {}: '.format(epoch))
            s_time = time.time()
            batch_loss, batch_loss_a, batch_loss_v, batch_loss_a_p, batch_loss_v_p, a_angle, v_angle, ratio_a, ratio_a_p, \
               a_diff, v_diff = train_epoch(args, epoch, model, device, train_dataloader, optimizer, scheduler,
                              audio_proto, visual_proto, writer, logger)
            audio_proto, visual_proto = calculate_prototype(args, model, train_dataloader, device, epoch, audio_proto, visual_proto)
            e_time = time.time()
            logger.info(f'per epoch time: {e_time-s_time}')
            # print('proto22', audio_proto[22], visual_proto[22])
            acc, acc_a, acc_v, acc_a_p, acc_v_p, loss_val = valid(args, model, device, val_dataloader, audio_proto, visual_proto, epoch, writer, logger)
            # logger.info('epoch: ', epoch, 'loss: ', batch_loss, batch_loss_a_p, batch_loss_v_p)
            # logger.info('epoch: ', epoch, 'acc: ', acc, 'acc_v_p: ', acc_v_p, 'acc_a_p: ', acc_a_p)

            if acc > best_acc:
                best_acc = float(acc)
                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)
                model_name = 'best_model_{}_of_{}_epoch{}_batch{}_lr{}_alpha{}.pth'.format(
                    args.method, args.optimizer, args.epochs, args.batch_size, args.learning_rate, args.alpha)

                saved_dict = {'saved_epoch': epoch,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)
                torch.save(saved_dict, save_dir)
                logger.info('The best model has been saved at {}.'.format(save_dir))
                logger.info("Loss: {:.4f}, Acc: {:.4f}, Acc_f: {:.4f}, Acc_v: {:.4f}, vloss: {:.4f}".format(
                    batch_loss, acc, acc_a, acc_v, loss_val))
            else:
                logger.info(
                    "Loss: {:.4f}, Acc: {:.4f}, Acc_f: {:.4f}, Acc_v: {:.4f},Best Acc: {:.4f}, vloss: {:.4f}".format(
                        batch_loss, acc, acc_a, acc_v, best_acc, loss_val))

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
