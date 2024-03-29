from torch.utils.tensorboard import SummaryWriter
import os
import torch
from dataset.av_dataset import AV_KS_Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from models.OGM.OGM_CD import CramedDataset

def ts_init(args):
    writer_path = os.path.join(args.tensorboard_path)
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)
    log_name = '{}_{}_{}_{}_epochs{}_batch{}_lr{}_alpha{}'.format(args.optimizer,  args.dataset, args.fusion_method, args.model, args.epochs, args.batch_size, args.learning_rate, args.alpha)
    writer = SummaryWriter(os.path.join(writer_path, log_name))
    return writer

def scalars_add(writer, epoch, batch_loss, val_loss, loss_a, loss_v, acc,acc_a, acc_v):
    writer.add_scalars('Loss', {'Total Loss': batch_loss, 'Val Loss': val_loss, 'Audio Loss':loss_a, 'Visual Loss':loss_v}, epoch)

    writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                            'Audio Accuracy': acc_a,
                                            'Visual Accuracy': acc_v}, epoch)
    return writer
    
def save_model(args, acc, epoch, model_dict, optimizer_dict, scheduler_dict,paras):
    if not os.path.exists(args.ckpt_path):
        os.mkdir(args.ckpt_path)
    model_name = '{}_best_model_of_{}_opti_{}_batch_{}_lr_{}_'.format(args.method, args.dataset,args.optimizer, args.batch_size, args.learning_rate)
    for x,y in paras.items():
        model_name+='{}_{}_'.format(x,y)
    model_name+='.pth'

    saved_dict = {'saved_epoch': epoch,
                    'fusion': args.fusion_method,
                    'acc': acc,
                    'model': model_dict,
                    'optimizer': optimizer_dict,
                    'scheduler': scheduler_dict}

    save_dir = os.path.join(args.ckpt_path, model_name)

    torch.save(saved_dict, save_dir)
    return save_dir

def train_performance(best_acc, acc_a, acc_v, batch_loss, valid_loss, args, acc, epoch, model_dict, optimizer_dict, scheduler_dict,paras):
    if acc > best_acc:
        best_acc = float(acc)

        save_dir = save_model(args, acc, epoch, model_dict, optimizer_dict, scheduler_dict,paras)

        print('The best model has been saved at {}.'.format(save_dir))
        print("Train Loss: {:.3f}, Valid Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, valid_loss, acc))
        print("Audio Acc: {:.3f}, Visual Acc: {:.3f} ".format(acc_a, acc_v))
    else:
        print("Train Loss: {:.3f}, Valid Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, valid_loss, acc, best_acc))
        print("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))
    return best_acc

def Dataloader_build(args, numworkers = 16):
    if args.dataset == 'KineticSound':
      train_dataset = AV_KS_Dataset(mode='train')
      valid_dataset = AV_KS_Dataset(mode='val')
      test_dataset = AV_KS_Dataset(mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(mode='train')
        test_dataset = CramedDataset(mode='test')
        valid_dataset = CramedDataset(mode ='val')
    elif args.dataset == 'AVE':
        train_dataset = AV_KS_Dataset(mode='train')
        valid_dataset = AV_KS_Dataset(mode='val')
        test_dataset = AV_KS_Dataset(mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=16, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=numworkers)
    val_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=numworkers)
    
    return train_dataloader, test_dataloader, val_dataloader

def Optimizer_build(args, model):
    if args.method == 'CKF':
        if args.optimizer == 'sgd':
            optimizer = optim.SGD([param for param in model.parameters() if param is not model.alpha], lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
            optimizer_alpha = optim.SGD([model.alpha], lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam([param for param in model.parameters() if param is not model.alpha], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
            optimizer_alpha = optim.Adam([model.alpha], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

        return optimizer, optimizer_alpha, scheduler

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=1e-4, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    return optimizer, scheduler