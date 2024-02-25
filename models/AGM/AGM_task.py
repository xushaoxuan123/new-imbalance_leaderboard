
from torch.optim import SGD,AdamW,Adam,lr_scheduler
from dataset.av_dataset import AV_KS_Dataset
from models.models import GradMod
from torch.utils.data import DataLoader


AGM_CONFIG = {
    'fusion':'concat',
    'fusion_type': 'late_fusion',
    'grad_norm_clip':0,
    'mode':'train',
    "threshold":0.099,
    "LAMBDA":100,
    'expt_name': 'AGM',
    'lr_scalar': 'lrstep',
    'methods': 'AGM',
}

class AGM_Config():
    def __init__(self, args):
        args_dict = vars(args)
        self.add_args(args_dict)
        self.select_model_params()
    def add_args(self,args_dict):
        for arg in args_dict.keys():
            setattr(self,arg,args_dict[arg])
    def select_model_params(self):
        self.add_args(AGM_CONFIG)


class AGM_task(object):
    def __init__(self,cfgs) -> None:
        super(AGM_task,self).__init__()
        self.cfgs = cfgs
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = self.build_loader()
        self.model = self.build_model()
        self.optimizer,self.scheduler = self.build_optimizer()

    def build_loader(self):
        train_dataset = AV_KS_Dataset(mode="train")
        train_dataloader = DataLoader(train_dataset, batch_size=self.cfgs.batch_size,
                                  shuffle=True, pin_memory=False)
        valid_dataloader = DataLoader(AV_KS_Dataset(mode="val"), batch_size=self.cfgs.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True)

        test_dataloader = DataLoader(AV_KS_Dataset(mode="test"), batch_size=self.cfgs.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True)
        return train_dataloader, valid_dataloader, test_dataloader
    
    def build_model(self):
        model = GradMod(self.cfgs)
        return model

    def build_optimizer(self):
        if self.cfgs.optimizer == 'sgd':
            optimizer = SGD(self.model.parameters(),lr=self.cfgs.learning_rate,momentum=self.cfgs.momentum, weight_decay=self.cfgs.weight_decay)
        elif self.cfgs.optimizer == 'adamw':
            optimizer = AdamW(self.model.parameters(),lr=self.cfgs.learning_rate,weight_decay=1e-4)
        elif self.cfgs.optimizer == 'adam':
            optimizer = Adam(self.model.parameters(),lr=self.cfgs.learning_rate)

        if self.cfgs.lr_scalar == 'lrstep':
            scheduler = lr_scheduler.StepLR(optimizer,self.cfgs.lr_decay_step,self.cfgs.lr_decay_ratio)
        elif self.cfgs.lr_scalar == 'cosinestep':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,eta_min=1e-6,last_epoch=-1)
        elif self.cfgs.lr_scalar == 'cosinestepwarmup':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,eta_min=1e-6,last_epoch=-1)
        return optimizer,scheduler