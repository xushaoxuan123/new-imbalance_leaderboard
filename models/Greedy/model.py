import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from models.backbone import resnet18 
import glob
import gin
from gin.config import _CONFIG

from models.Greedy.balanced_mmtm import MMTM_mitigate as MMTM
from models.Greedy.balanced_mmtm import get_rescale_weights

@gin.configurable
class MMTM_MVCNN(nn.Module):
    def __init__(self, 
        nclasses=31,#n_classes 
        num_views=2, 
        pretraining=False,
        mmtm_off=False,
        mmtm_rescale_eval_file_path = None,
        mmtm_rescale_training_file_path = None,
        device='cuda:0',
        saving_mmtm_scales=False,
        saving_mmtm_squeeze_array=False,
        ):
        super(MMTM_MVCNN, self).__init__()

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.num_views = num_views

        self.mmtm_off = mmtm_off
        if self.mmtm_off:
            self.mmtm_rescale = get_rescale_weights(
                mmtm_rescale_eval_file_path, 
                mmtm_rescale_training_file_path,
                validation=False, 
                starting_mmtmindice = 1, 
                mmtmpositions=4,
                device=torch.device(device),
            )
        self.device = device
        self.saving_mmtm_scales = saving_mmtm_scales
        self.saving_mmtm_squeeze_array = saving_mmtm_squeeze_array

        # self.net_view_0 = models.resnet18(pretrained=pretraining)
        # self.net_view_0.fc = nn.Linear(512, nclasses)
        # self.net_view_1 = models.resnet18(pretrained=pretraining)
        # self.net_view_1.fc = nn.Linear(512, nclasses)

        self.net_view_0 = resnet18(modality = 'visual')
        self.net_view_0.fc = nn.Linear(512, nclasses)
        self.net_view_1 = resnet18(modality= 'audio')
        self.net_view_1.fc = nn.Linear(512, nclasses)

        self.mmtm2 = MMTM(128, 128, 4)
        self.mmtm3 = MMTM(256, 256, 4)
        self.mmtm4 = MMTM(512, 512, 4)


    def forward(self, image, spec, curation_mode=False, caring_modality=None):
        image = image.permute(0, 2, 1, 3, 4).contiguous()
        (B, T, C, H, W) = image.size()
        image = image.view(B * T, C, H, W)
        image = image.to(self.device)
        spec = spec.to(self.device)
        avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        ## KS_dataset
        frames_view_0 = self.net_view_0.conv1(image)
        frames_view_0 = self.net_view_0.bn1(frames_view_0)
        frames_view_0 = self.net_view_0.relu(frames_view_0)
        frames_view_0 = self.net_view_0.maxpool(frames_view_0)

        frames_view_1 = self.net_view_1.conv1(spec)
        frames_view_1 = self.net_view_1.bn1(frames_view_1)
        frames_view_1 = self.net_view_1.relu(frames_view_1)
        frames_view_1 = self.net_view_1.maxpool(frames_view_1)

        frames_view_0 = self.net_view_0.layer1(frames_view_0)
        frames_view_1 = self.net_view_1.layer1(frames_view_1)

        scales = []
        squeezed_mps = [] 

        for i in [2, 3, 4]:

            frames_view_0 = getattr(self.net_view_0, f'layer{i}')(frames_view_0)
            frames_view_1 = getattr(self.net_view_1, f'layer{i}')(frames_view_1)

            frames_view_0, frames_view_1, scale, squeezed_mp = getattr(self, f'mmtm{i}')(
                frames_view_0, 
                frames_view_1, 
                self.saving_mmtm_scales,
                self.saving_mmtm_squeeze_array,
                turnoff_cross_modal_flow = True if self.mmtm_off else False,
                average_squeezemaps = self.mmtm_rescale[i-1] if self.mmtm_off else None,
                curation_mode = curation_mode,
                caring_modality = caring_modality
                )
            scales.append(scale)
            squeezed_mps.append(squeezed_mp)
        (_, C, H, W) = frames_view_0.size()
        B = frames_view_1.size()[0]
        frames_view_0 = frames_view_0.view(B, -1, C, H, W)
        frames_view_0 = frames_view_0.permute(0, 2, 1, 3, 4)
        frames_view_0 = F.adaptive_avg_pool3d(frames_view_0,1)
        frames_view_1 = F.adaptive_avg_pool2d(frames_view_1,1)
        
        x_0 = torch.flatten(frames_view_0, 1)
        x_0 = self.net_view_0.fc(x_0)

        x_1 = torch.flatten(frames_view_1, 1)
        x_1 = self.net_view_1.fc(x_1)
        
        return (x_0+x_1)/2, [x_0, x_1], scales, squeezed_mps



