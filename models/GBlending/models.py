import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models import ConcatFusion
from models.backbone import resnet18

class AVClassifier_gb(nn.Module):
    def __init__(self, args):
        super(AVClassifier_gb, self).__init__()

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

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

        self.fc_a = nn.Linear(512, n_classes)
        self.fc_v = nn.Linear(512, n_classes)
        self.fc_out = nn.Linear(1024, n_classes)

    def forward(self, audio=None, visual=None, types=0):
        if types == 0:
            visual = visual.permute(0, 2, 1, 3, 4).contiguous()
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
            out = torch.cat((a, v), 1)

            out_a = self.fc_a(a)
            out_v = self.fc_v(v)
            out = self.fc_out(out)

            return out_a, out_v, out

        elif types == 1:
            a = self.audio_net(audio)
            a = F.adaptive_avg_pool2d(a, 1)
            a = torch.flatten(a, 1)
            out_a = self.fc_a(a)

            return out_a

        else:
            B = visual.size()[0]
            visual = visual.permute(0, 2, 1, 3, 4).contiguous()
            v = self.visual_net(visual)

            (_, C, H, W) = v.size()
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            v = F.adaptive_avg_pool3d(v, 1)
            v = torch.flatten(v, 1)

            out_v = self.fc_v(v)
            return out_v


class RFClassifier_gb(nn.Module):
    def __init__(self, args):
        super(RFClassifier_gb, self).__init__()

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'UCF-101':
            n_classes = 101
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        self.fusion_module = ConcatFusion(output_dim=n_classes)

        self.flow_net = resnet18(modality='flow')
        state = torch.load('/home/ruoxuan_feng/models/resnet18.pth')
        del state['conv1.weight']
        self.flow_net.load_state_dict(state, strict=False)
        self.visual_net = resnet18(modality='visual')
        self.visual_net.load_state_dict(torch.load('/home/ruoxuan_feng/models/resnet18.pth'), strict=False)

        self.fc_f = nn.Linear(512, n_classes)
        self.fc_v = nn.Linear(512, n_classes)
        self.fc_out = nn.Linear(1024, n_classes)

    # type = 0 both  1 audio/flow  2visual
    def forward(self, flow=None, visual=None, drop=None, drop_arg=None, types=0):
        if types == 0:
            B = visual.size()[0]
            f = self.flow_net(flow)
            v = self.visual_net(visual)

            (_, C, H, W) = v.size()
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            (_, C, H, W) = f.size()
            f = f.view(B, -1, C, H, W)
            f = f.permute(0, 2, 1, 3, 4)

            f = F.adaptive_avg_pool3d(f, 1)
            v = F.adaptive_avg_pool3d(v, 1)

            f = torch.flatten(f, 1)
            v = torch.flatten(v, 1)
            out = torch.cat((f, v), 1)

            out_f = self.fc_f(f)
            out_v = self.fc_v(v)
            out = self.fc_out(out)

            return out_f, out_v, out

        elif types == 1:
            B = flow.size()[0]
            # visual = visual.permute(0, 2, 1, 3, 4).contiguous()
            f = self.flow_net(flow)

            (_, C, H, W) = f.size()
            f = f.view(B, -1, C, H, W)
            f = f.permute(0, 2, 1, 3, 4)

            f = F.adaptive_avg_pool3d(f, 1)
            f = torch.flatten(f, 1)

            out_f = self.fc_f(f)
            return out_f

        else:
            B = visual.size()[0]
            # visual = visual.permute(0, 2, 1, 3, 4).contiguous()
            v = self.visual_net(visual)

            (_, C, H, W) = v.size()
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            v = F.adaptive_avg_pool3d(v, 1)
            v = torch.flatten(v, 1)

            out_v = self.fc_v(v)
            return out_v