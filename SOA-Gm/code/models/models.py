import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import resnet18


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
        self.visual_net.load_state_dict(torch.load(''), strict=False)
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
        state = torch.load('')
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
        state = torch.load('')
        del state['conv1.weight']
        self.flow_net.load_state_dict(state, strict=False)
        print('load pretrain')
        self.visual_net = resnet18(modality='visual')
        self.visual_net.load_state_dict(torch.load(''), strict=False)
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

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == 'UFC101':
            n_classes = 101
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))


        self.dataset = args.dataset
        args.pretrained = True
        self.pretrained = args.pretrained
        if args.dataset == 'UFC101':
            self.audio_net = resnet18(modality='visual')
            self.audio_net.load_state_dict(torch.load(''), strict=False)

        else:
            if args.pretrained:
                self.audio_net = resnet18(modality='audio')
                state = torch.load('')
                state = state['model']
                self.audio_net.load_state_dict(state, strict=False)

            else:
                self.audio_net = resnet18(modality='audio')

        self.visual_net = resnet18(modality='visual')
        if args.dataset == 'UFC101':
            self.visual_net.load_state_dict(torch.load(''), strict=False)
        if args.pretrained:
            dict = torch.load('')
            state = dict['model']
            self.visual_net.load_state_dict(state, strict=False)

        self.head = nn.Linear(1024, n_classes)
        self.head_audio = nn.Linear(512, n_classes)
        self.head_video = nn.Linear(512, n_classes)

        if args.pretrained:
            for param in self.audio_net.parameters():
                param.requires_grad = True

            # 冻结 visual_net 的参数
            for param in self.visual_net.parameters():
                param.requires_grad = True


    def forward(self, audio, visual):
        if self.dataset not in ['CREMAD','UFC101']:
            visual = visual.permute(0, 2, 1, 3, 4).contiguous()
        if audio.dim()==3:
            audio = audio.unsqueeze(1)
        a = self.audio_net(audio)
        v = self.visual_net(visual)
        (_, C, H, W) = v.size()
        B = visual.size()[0]
        if self.dataset in ['UFC101']:
            a = a.view(B,-1,C,H,W)
            a = a.permute(0,2,1,3,4)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        if self.dataset in ['UFC101']:
            a = F.adaptive_avg_pool3d(a,1)
        else:
            a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)


        out = torch.cat((a,v),1)
        out = self.head(out)

        out_audio=self.head_audio(a)
        out_video=self.head_video(v)

        return out,out_audio,out_video

class AVClassifier_AGM(nn.Module):
    def __init__(self, args):
        super(AVClassifier_AGM, self).__init__()

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == 'UFC101':
            n_classes = 101
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        args.pretrained = False
        self.dataset = args.dataset
        if args.dataset == 'UFC101':
            self.audio_net = resnet18(modality='visual')
            self.audio_net.load_state_dict(torch.load(''), strict=False)

        else:
            if args.pretrained:
                self.audio_net = resnet18(modality='audio')
                state = torch.load('')
                state = state['model']
                self.audio_net.load_state_dict(state, strict=False)

            else:
                self.audio_net = resnet18(modality='audio')

        self.visual_net = resnet18(modality='visual')
        if args.dataset == 'UFC101':
            self.visual_net.load_state_dict(torch.load(''), strict=False)
        if args.pretrained:
            dict = torch.load('')
            state = dict['model']
            self.visual_net.load_state_dict(state, strict=False)

        self.head = nn.Linear(1024, n_classes)
        self.head_audio = nn.Linear(512, n_classes)
        self.head_video = nn.Linear(512, n_classes)

        if args.pretrained:
            for param in self.audio_net.parameters():
                param.requires_grad = False

            # 冻结 visual_net 的参数
            for param in self.visual_net.parameters():
                param.requires_grad = False


    def forward(self, audio, visual,pad_a=False,pad_v=False):
        if self.dataset not in ['CREMAD','UFC101']:
            visual = visual.permute(0, 2, 1, 3, 4).contiguous()
        if audio.dim()==3:
            audio = audio.unsqueeze(1)

        a = self.audio_net(audio)
        # print(visual.size())
        v = self.visual_net(visual)

        if pad_a:
            a = torch.zeros_like(a,device=a.device)
        if pad_v:
            v = torch.zeros_like(v,device=v.device)

        (_, C, H, W) = v.size()
        B = visual.size()[0]
        if self.dataset in ['UFC101']:
            a = a.view(B,-1,C,H,W)
            a = a.permute(0,2,1,3,4)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        if self.dataset in ['UFC101']:
            a = F.adaptive_avg_pool3d(a,1)
        else:
            a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)


        out = torch.cat((a,v),1)
        out = self.head(out)

        out_audio=self.head_audio(a)
        out_video=self.head_video(v)

        return out


class Modality_Visual(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,total_out,pad_visual_out,pad_audio_out):
        return 0.5*(total_out-pad_visual_out+pad_audio_out)

class Modality_Audio(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,total_out,pad_visual_out,pad_audio_out):
        return 0.5*(total_out-pad_audio_out+pad_visual_out)

class Modality_out(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x


class AGM(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.net = AVClassifier_AGM(args)
        self.m_v = Modality_Visual()
        self.m_a = Modality_Audio()
        self.m_v_o = Modality_out()
        self.m_a_o = Modality_out()
        self.scale_a = 1.0
        self.scale_v = 1.0

        # self.mode = mode

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

    def forward(self,audio,visual,mode='train'):
        total_out = self.net(audio,visual,pad_a=False,pad_v=False)
        self.net.eval()
        pad_visual_out = self.net(audio,visual,pad_a = False,pad_v=True)
        pad_audio_out= self.net(audio, visual, pad_a=True, pad_v=False)
        zero_padding_out= self.net(audio, visual, pad_a=True, pad_v=True)
        if mode=="train":
            self.net.train()
        m_a = self.m_a_o(self.m_a(total_out, pad_visual_out, pad_audio_out))
        m_v = self.m_v_o(self.m_v(total_out, pad_visual_out, pad_audio_out))
        c = total_out - pad_visual_out - pad_audio_out + zero_padding_out
        return m_a, m_v, m_a + m_v, c

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


class AClassifier(nn.Module):
    def __init__(self, args):
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

        self.net = resnet18(modality='audio')
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, audio):
        a = self.net(audio)
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)
        out = self.classifier(a)
        return out


class VClassifier(nn.Module):
    def __init__(self, args):
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

        self.net = resnet18(modality='visual')
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, visual):
        B = visual.size(0)
        visual = visual.permute(0, 2, 1, 3, 4).contiguous()
        v = self.net(visual)
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        out = self.classifier(v)
        return out

        
    

        
    




