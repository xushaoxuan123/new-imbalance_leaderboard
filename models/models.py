import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusions.fusion_method import SumFusion, ConcatFusion, FiLM, GatedFusion
from .backbone import resnet18
from models.module_base import MaxOut_MLP, MLP

class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024+512, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, out):
        # output = torch.cat((x, y), dim=1)
        output = self.fc_out(out)
        return output



class RGBClassifier(nn.Module):
    def __init__(self, args):
        super(RGBClassifier, self).__init__()

        n_classes = 101

        self.visual_net = resnet18(modality='visual')
        self.visual_net.load_state_dict(torch.load('/home/yake_wei/models/resnet18.pth'), strict=False)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, visual):
        B = visual.size()[0]
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        v = F.adaptive_avg_pool3d(v, 1)

        v = torch.flatten(v, 1)

        out = self.fc(v)

        return out

class FlowClassifier(nn.Module):
    def __init__(self, args):
        super(FlowClassifier, self).__init__()

        n_classes = 101

        self.flow_net = resnet18(modality='flow')
        state = torch.load('/home/yake_wei/models/resnet18.pth')
        del state['conv1.weight']
        self.flow_net.load_state_dict(state, strict=False)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, flow):
        B = flow.size()[0]
        v = self.flow_net(flow)

        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        v = F.adaptive_avg_pool3d(v, 1)

        v = torch.flatten(v, 1)

        out = self.fc(v)

        return out

class RFClassifier(nn.Module):
    def __init__(self, args):
        super(RFClassifier, self).__init__()

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'UCF101':
            n_classes = 101
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        self.flow_net = resnet18(modality='flow')
        state = torch.load('/home/yake_wei/models/resnet18.pth')
        del state['conv1.weight']
        self.flow_net.load_state_dict(state, strict=False)
        print('load pretrain')
        self.visual_net = resnet18(modality='visual')
        self.visual_net.load_state_dict(torch.load('/home/yake_wei/models/resnet18.pth'), strict=False)
        print('load pretrain')

        self.head = nn.Linear(1024, n_classes)
        self.head_flow = nn.Linear(512, n_classes)
        self.head_video = nn.Linear(512, n_classes)



    def forward(self, flow, visual):
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

        
        out = torch.cat((f,v),1)
        out = self.head(out)

        out_flow=self.head_flow(f)
        out_video=self.head_video(v)

        return out,out_flow,out_video


class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()
        self.method = args.method

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


        self.dataset = args.dataset
        self.alpha = nn.Parameter(torch.tensor(0.5))  ## used for CKF to learn
        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

        self.head = nn.Linear(1024, n_classes)
        self.head_audio = nn.Linear(512, n_classes)
        self.head_video = nn.Linear(512, n_classes)



    def forward(self, audio, visual):
        if self.dataset != 'CREMAD':
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
        if self.method == 'CKF':
            out = self.alpha * self.head_audio(a) +(1- self.alpha) * self.head_video(v)
            return a, v, out
        out = torch.cat((a,v),1)
        out = self.head(out)

        if self.method == 'MMCosine' or self.method == 'CML' or self.method == 'MSBD' or self.method == 'UNM':
            return out, a, v
        if self.method == 'PMR':
            return a, v, out

        out_audio = self.head_audio(a)
        out_video = self.head_video(v)

        return out, out_audio, out_video

class AClassifier(nn.Module):
    def __init__(self,args):
        super(AClassifier, self).__init__()

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
        self.dataset = args.dataset

        self.audio_net = resnet18(modality='audio')
        self.head_audio = nn.Linear(512, n_classes)
    
    def forward(self, audio):
        a = self.audio_net(audio)
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)
        out = self.head_audio(a)
        return out ,a
        
class VClassifier(nn.Module):
    def __init__(self,args):
        super(VClassifier, self).__init__()

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
        self.dataset = args.dataset
        self.args = args
        self.visual_net = resnet18(modality='visual')
        self.head_visual = nn.Linear(512, n_classes)
    
    def forward(self, visual):
        if self.dataset != 'CREMAD':
            visual = visual.permute(0, 2, 1, 3, 4).contiguous()
       
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = int(v.size()[0]/3)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        v = F.adaptive_avg_pool3d(v, 1)
 
        v = torch.flatten(v, 1)

        out = self.head_visual(v)
        return out ,v
    

class AVClassifier_OGM(nn.Module):
    def __init__(self, args):
        super(AVClassifier_OGM, self).__init__()

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


        self.dataset = args.dataset

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

        self.head = nn.Linear(1024, n_classes)
        self.head_audio = nn.Linear(512, n_classes)
        self.head_video = nn.Linear(512, n_classes)

    def forward(self, audio, visual):
        if self.dataset != 'CREMAD':
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

        out = torch.cat((a,v),1)
        out = self.head(out)

        return a,v,out
    
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


        self.fusion_module = SumFusion(output_dim=n_classes)
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
        co = self.fusion_module(a,v)
        out_a = self.linear_star(a)
        out_v = self.linear_star(v)
        out_co = self.linear_star(co)


        return _a,_v,out_a,out_v,out_co
    
class Modality_Visual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, total_out, pad_visual_out, pad_audio_out):
        return 0.5 * (total_out - pad_visual_out + pad_audio_out)


class Modality_Audio(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, total_out, pad_visual_out, pad_audio_out):
        return 0.5 * (total_out - pad_audio_out + pad_visual_out)

class AV_Early_Classifier(nn.Module):
    """Using Resnet as muliti-modal encoder for early-fusion model.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, args):
        super(AV_Early_Classifier, self).__init__()
        self.mode = "classify"
        self.args = args
        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')
        self.mm_encoder = MaxOut_MLP(6, 2048, 1024, 1024, linear_layer=False)
        self.head = MLP(1024, 128, 6, one_layer=True)

    def forward(self, audio, visual, pad_audio=False, pad_visual=False):
        if pad_audio:
            audio = torch.zeros_like(audio, device=audio.device)

        if pad_visual:
            visual = torch.zeros_like(visual, device=visual.device)

        a = self.audio_net(audio)
        v = self.visual_net(visual)
        (T, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)
        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        encoded_out = torch.cat([a, v], dim=1)
        feature = self.mm_encoder(encoded_out)
        out = self.head(feature)
        if self.mode == "feature":
            return out, feature
        return out
    
class AV_Late_Classifier(nn.Module):
    """Using Resnet as muliti-modal encoder for early-fusion model.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, args):
        super(AV_Late_Classifier, self).__init__()
        self.mode = "classify"
        self.args = args
        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')
        self.mm_encoder = MaxOut_MLP(6, 2048, 1024, 1024, linear_layer=False)
        # self.head = MLP(1024, 128, 6, one_layer=True)
        self.head = MLP(1024, 128, 31, one_layer=True)

    def forward(self, audio, visual, pad_audio=False, pad_visual=False):
        if pad_audio:
            audio = torch.zeros_like(audio, device=audio.device)

        if pad_visual:
            visual = torch.zeros_like(visual, device=visual.device)

        a = self.audio_net(audio)
        v = self.visual_net(visual)
        (T, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)
        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        encoded_out = torch.cat([a, v], dim=1)
        feature = self.mm_encoder(encoded_out)
        out = self.head(feature)
        if self.mode == "feature":
            return out, feature
        return out

class Modality_out(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class GradMod(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.mode = cfgs.mode
        self.extract_mm_feature = False
        if cfgs.fusion_type == 'late_fusion': #early
            self.net = AV_Late_Classifier(cfgs)
        elif cfgs.fusion_type == 'early_fusion':
            self.net = AV_Early_Classifier(cfgs)
        self.m_v = Modality_Visual()
        self.m_a = Modality_Audio()
        self.m_v_o = Modality_out()
        self.m_a_o = Modality_out()

        self.scale_a = 1.0
        self.scale_v = 1.0

        self.m_a_o.register_full_backward_hook(self.hooka)
        self.m_v_o.register_full_backward_hook(self.hookv)

    def hooka(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_a,

    def hookv(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_v,

    def update_scale(self, coeff_a, coeff_v):
        self.scale_a = coeff_a
        self.scale_v = coeff_v

    def forward(self, audio, visual):
        self.net.mode = "feature"
        total_out, encoded_feature = self.net(audio, visual, pad_audio=False, pad_visual=False)
        # print(f'2.1 cuda allocated: {torch.cuda.memory_allocated() // (1024 ** 3)} GB')

        self.net.mode = "classify"
        self.net.eval()
        with torch.no_grad():
            pad_visual_out = self.net(audio, visual, pad_audio=False, pad_visual=True)
            pad_audio_out = self.net(audio, visual, pad_audio=True, pad_visual=False)
            zero_padding_out = self.net(audio, visual, pad_audio=True, pad_visual=True)
        # print(f'2.2 cuda allocated: {torch.cuda.memory_allocated() // (1024 ** 3)} GB')

        if self.mode == "train":
            self.net.train()
        m_a = self.m_a_o(self.m_a(total_out, pad_visual_out, pad_audio_out))
        m_v = self.m_v_o(self.m_v(total_out, pad_visual_out, pad_audio_out))
        # print(f'2.3 cuda allocated: {torch.cuda.memory_allocated() // (1024 ** 3)} GB')

        if self.extract_mm_feature is True:
            return total_out, pad_visual_out, pad_audio_out, zero_padding_out, m_a + m_v, encoded_feature
        return total_out, pad_visual_out, pad_audio_out, zero_padding_out, m_a + m_v
    

