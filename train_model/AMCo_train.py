from train_model.support import ts_init, scalars_add, train_performance, Dataloader_build, Optimizer_build
import torch
from models.models import AVClassifier_ACMo 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.models import AVClassifier_gb, RFClassifier_gb
from dataset.av_dataset import AV_KS_Dataset
from models.OGM.OGM_CD import CramedDataset#这里还没改，之后再改CrameD
from utils.utils import setup_seed, weight_init
from models.ACMo.ACMo_main import train_epoch, valid
import os
def ACMo_main(args):
    #print(args)
    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    model = AVClassifier_ACMo(args)

    model.apply(weight_init)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda()

    optimizer, scheduler = Optimizer_build(args, model)

    # if args.dataset == 'VGGSound':
    #     train_dataset = VGGSound(args, mode='train')
    #     test_dataset = VGGSound(args, mode='test')
    train_dataloader, test_dataloader, valid_dataloader = DataLoader(args)
    

    if args.train:

        best_acc = 0.0
        l_t = 0
        if args.use_tensorboard:
            writer = ts_init()
        for epoch in range(args.epochs):
            
            print('Epoch: {}: '.format(epoch))
            
            l_t+=epoch/10
            batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                        train_dataloader, optimizer, scheduler, l_t,writer)
            acc, acc_a, acc_v, valid_loss = valid(args, model, device, test_dataloader)
            if args.use_tensorboard:
                scalars_add(writer, epoch, batch_loss, valid_loss, batch_loss_a, batch_loss_v, acc, acc_a, acc_v)

            best_acc = train_performance(best_acc, acc_a, acc_v, batch_loss, valid_loss, args, acc, epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),{'alpha':args.alpha, 'U':args.U, 'sigma':args.sigma, 'epsilon':args.eps})
        writer.close()

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

        acc, acc_a, acc_v, _ = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))