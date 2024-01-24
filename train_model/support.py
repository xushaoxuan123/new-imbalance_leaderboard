from torch.utils.tensorboard import SummaryWriter
import os
import torch
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