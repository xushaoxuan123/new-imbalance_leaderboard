import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb

from models.OGM.OGM_CD import CramedDataset#这里还没改，之后再改CrameD
from dataset.av_dataset import AV_KS_Dataset as AVDataset
from models.ACMo.ACMo_AVC import AVClassifier_ACMo as AVClassifier
from utils.utils import setup_seed, weight_init


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler,l_t, writer=None):##三个参数
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()
    print("Start training ... ")
    U=args.U
    sigma = args.sigma # args.sigma
    eps = args.eps # args.eps
    pt = np.sin(np.pi/2*(min(eps,l_t)/eps))
    N = int(pt * U)
    mask_t = np.ones(U-N)
    mask_t = np.pad(mask_t,(0,N))
    np.random.shuffle(mask_t)
    mask_t = torch.from_numpy(mask_t)
    mask_t = mask_t.to(device)
    mask_none = np.ones(U)
    mask_none = torch.from_numpy(mask_none)
    mask_none = mask_none.to(device)
    _loss = 0
    _loss_a = 0
    _loss_v = 0
    _out_a = 0
    _out_v = 0
    _out_co = 0
    dependent_modality = 'none'
    sft_oa=0
    sft_ov=0
    for step, (image, spec, label) in enumerate(dataloader):

        #pdb.set_trace()
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        if args.modulation_starts<= epoch <= args.modulation_ends:
            _, _, out_a, out_v, out_co = model(spec.unsqueeze(1).float(), image.float(), mask_t ,dependent_modality,pt)
        else:
            _, _, out_a, out_v, out_co = model(spec.unsqueeze(1).float(), image.float(), mask_none ,'none',pt)

        # if args.fusion_method == 'sum':
        #     out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
        #              model.module.fusion_module.fc_y.bias)
        #     out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
        #              model.module.fusion_module.fc_x.bias)
        # else:
        #     weight_size = model.module.fusion_module.fc_out.weight.size(1)
        #     out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
        #              + model.module.fusion_module.fc_out.bias / 2)

        #     out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
        #              + model.module.fusion_module.fc_out.bias / 2)

        loss = criterion(out_co, label)
        loss_v = criterion(out_v, label)
        loss_a = criterion(out_a, label)
        loss.backward(retain_graph=True)
        loss_v.backward()
        loss_a.backward()
        


            # if args.modulation_starts <= epoch <= args.modulation_ends: # bug fixed
            #     for name, parms in model.named_parameters():
            #         layer = str(name).split('.')[1]

            #         if 'audio' in layer and len(parms.grad.size()) == 4:
            #             if args.fusion_method == 'OGM_GE':  # bug fixed
            #                 parms.grad = parms.grad * coeff_a + \
            #                              torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
            #             elif args.fusion_method == 'OGM':
            #                 parms.grad *= coeff_a

            #         if 'visual' in layer and len(parms.grad.size()) == 4:
            #             if args.fusion_method == 'OGM_GE':  # bug fixed
            #                 parms.grad = parms.grad * coeff_v + \
            #                              torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
            #             elif args.fusion_method == 'OGM':
            #                 parms.grad *= coeff_v
            # else:
            #     pass
        out_combine = torch.cat((out_a,out_v),1)
        sft_out = softmax(out_combine)
        sft_oa = torch.sum(sft_out[:,0:model.module.n_classes])/(args.batch_size)
        sft_ov = torch.sum(sft_out[:,model.module.n_classes:])/(args.batch_size)
        # print(sft_oa,sft_ov,sft_out.size())
        if(sft_oa>=sigma):
            dependent_modality ='audio'
        elif(sft_ov>=sigma):
            dependent_modality = 'visul'
        optimizer.step()
        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()
        # print(out_a.size())
        # _out_a += out_a
        
        # _out_v += out_v
        # _out_co += out_co
    scheduler.step()
  

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        criterion = nn.CrossEntropyLoss()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]
        mask_t = np.ones(args.U)
        valid_loss = 0 
        for step, (image, spec, label) in enumerate(dataloader):
            
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            a, v, a_out, v_out, out = model(spec.unsqueeze(1).float(), image.float(),mask_t,'0',0)

            # if args.fusion_method == 'sum':
            #     out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
            #              model.module.fusion_module.fc_y.bias / 2)
            #     out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
            #              model.module.fusion_module.fc_x.bias / 2)
            # else:
            #     out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
            #              model.module.fusion_module.fc_out.bias / 2)
            #     out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) +
            #              model.module.fusion_module.fc_out.bias / 2)
                

            prediction = softmax(out)
            pred_v = softmax(v_out)
            pred_a = softmax(a_out)
            loss = criterion(out, label)
            valid_loss += loss.item() #new
            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                #pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num), valid_loss/ len(dataloader)


def ACMo_main(args):
    #print(args)
    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    model = AVClassifier(args)

    model.apply(weight_init)
   

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    # if args.dataset == 'VGGSound':
    #     train_dataset = VGGSound(args, mode='train')
    #     test_dataset = VGGSound(args, mode='test')
    if args.dataset == 'KineticSound':
        train_dataset = AVDataset(mode='train')
        test_dataset = AVDataset(mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(mode='train')
        test_dataset = CramedDataset(mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVDataset(mode='train')
        test_dataset = AVDataset(mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)

    if args.train:

        best_acc = 0.0
        l_t = 0
        for epoch in range(args.epochs):
            
            print('Epoch: {}: '.format(epoch))
            
            l_t+=epoch/10
            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path, args.dataset)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_{}'.format(args.method, args.fusion_method)
                writer = SummaryWriter(os.path.join(writer_path, log_name))

                batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler, l_t,writer)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

                writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                            'Audio Loss': batch_loss_a,
                                            'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)

            else:
                batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler,l_t,writer)
                acc, acc_a, acc_v, valid_loss = valid(args, model, device, test_dataloader)

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = '{}_best_model_of_dataset_{}_{}_alpha_{}_' \
                             'optimizer_{}_modulate_starts_{}_ends_{}_U={}_sigma{}_eps{}' \
                             '.pth'.format(args.method,args.dataset,
                                                          args.fusion_method,
                                                          args.alpha,
                                                          args.optimizer,
                                                          args.modulation_starts,
                                                          args.modulation_ends,
                                                          args.U,
                                                          args.sigma,
                                                          args.eps
                                                          )

                saved_dict = {'saved_epoch': epoch,
                              'modulation': args.fusion_method,
                              'alpha': args.alpha,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
            else:
                print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))

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

        assert modulation == args.fusion_method, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model = model.load_state_dict(state_dict)
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


