import torch
import torch.nn as nn
import numpy as np
import random
import logging
import os

class MethodSettings():
    def __init__(self, method_name):
        self.method_name = method_name

    def reset(self):
        self.loss_ratio = 0
        self.metric1 = 0
        self.metric2 = 0

    def update(self, ):
        pass



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def re_init(module):
    for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)


def get_logger(logger_name, logger_dir=None, log_name=None, is_mute_logger=False):
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()

    if is_mute_logger:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    hterm = logging.StreamHandler()
    hterm.setFormatter(formatter)
    hterm.setLevel(logging.INFO)
    logger.addHandler(hterm)

    if logger_dir is not None:
        if log_name is None:
            logger_path = os.path.join(logger_dir, f"{logger_name}.log")
        else:
            logger_path = os.path.join(logger_dir, log_name)
        hfile = logging.FileHandler(logger_path)
        hfile.setFormatter(formatter)
        hfile.setLevel(logging.INFO)
        logger.addHandler(hfile)
    return logger


def accuracy(logits, target):
    logits = logits.detach().cpu()
    target = target.detach().cpu()

    preds = logits.argmax(dim=-1)
    assert preds.shape == target.shape
    correct = torch.sum(preds==target)
    total = torch.numel(target)
    if total == 0:
        return 1
    else:
        return correct / total


def save_model_name(args):
    model_name = 'best_model_{}_of_{}_epoch{}_batch{}_lr{}_alpha{}.pth'.format(
        args.method, args.optimizer, args.epochs, args.batch_size, args.learning_rate,
        args.alpha)




