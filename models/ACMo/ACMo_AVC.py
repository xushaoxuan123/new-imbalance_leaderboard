import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import resnet18
from models.OGM.OGM_fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion


class AVClassifier_ACMo(nn.Module):
    def __init__(self, args):
        super(AVClassifier_ACMo, self).__init__()

        fusion = args.fusion_method
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

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')
        self.linear_a = nn.Linear(512,args.U)
        self.linear_v = nn.Linear(512,args.U)
        self.linear_star = nn.Linear(100,n_classes)#
        self.n_classes = n_classes

    def forward(self, audio, visual, masks ,depd_modality,pt):
        
        a = self.audio_net(audio)
        v = self.visual_net(visual)
        
        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)
        
        _a = a
        _v = v

        a = self.linear_a(a)
        v = self.linear_v(v)
        
        if depd_modality =="audio":
            a = torch.mul(a,masks)
            if(abs(pt-1)>0.1):
                a = a*1/(1-pt)
            else:
                a = a*10
        elif depd_modality == 'visual':
            v = torch.mul(v,masks)
            if(abs(pt-1)>0.1):
                v = v*1/(1-pt)
            else:
                v = v*10
        co = a+v
        a = a.to(torch.float32)
        v = v.to(torch.float32)
        co = co.to(torch.float32)
        out_a = self.linear_star(a)
        out_v = self.linear_star(v)
        out_co = self.linear_star(co)


        return _a,_v,out_a,out_v,out_co
