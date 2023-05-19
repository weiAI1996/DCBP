import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from .backbone import BN_MOMENTUM, hrnet_classification
import skimage.io as io

class HRnet_Backbone(nn.Module):
    def __init__(self, backbone = 'hrnetv2_w18', pretrained = False):
        super(HRnet_Backbone, self).__init__()
        self.model    = hrnet_classification(backbone = backbone, pretrained = pretrained)
        del self.model.incre_modules
        del self.model.downsamp_modules
        del self.model.final_layer
        del self.model.classifier
        

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)
        
        x_list = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list.append(self.model.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.model.stage2(x_list)

        x_list = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.model.transition2[i](y_list[i]))
                else:
                    x_list.append(self.model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage3(x_list)

        x_list = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.model.transition3[i](y_list[i]))
                else:
                    x_list.append(self.model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage4(x_list)
        
        return y_list

class scale_atten(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(scale_atten, self).__init__()
        self.ch_in = ch_in
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1) 
        return x * y.expand_as(x) 
    

class Local_Context(nn.Module):
    def __init__(self):
        super(Local_Context, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out = x*out
        return out
class Glob_Context(nn.Module):

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=2,
                 sub_sample=True,
                 bn_layer=True):
        super(Glob_Context, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)#[bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)#[bs, w*h/4, c]
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)#[bs, w*h, c]
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)#[bs, c, w*h/4]
        f = torch.matmul(theta_x, phi_x)#[bs, w*h, w*h/4]
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class DCBP(nn.Module):
    def __init__(self, num_classes = 21, backbone = 'hrnetv2_w18', pretrained = False):
        super(DCBP, self).__init__()
        self.backbone       = HRnet_Backbone(backbone = backbone, pretrained = pretrained)

        last_inp_channels   = np.int(np.sum(self.backbone.model.pre_stage_channels))
        self.num_classes = num_classes

        self.sobel_x = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).cuda()
        self.sobel_y = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).cuda()
        self.laplace = torch.tensor([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).cuda()

        self.scale_atten = scale_atten(448)
        self.gcm = Glob_Context(448)
        self.lcf=Local_Context()
        self.fuse_layer = nn.Sequential(
            nn.Conv2d(in_channels=448*2, out_channels=448, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(448, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.seg_last_layer = nn.Sequential(
            nn.Conv2d(in_channels=448, out_channels=448, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(448, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=448, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )
        self.line_fuse_layer = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        )
        self.line_last_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=3, out_channels=2, kernel_size=1, stride=1, padding=0)
        )
        self.last_layer = nn.Sequential(nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(num_classes, momentum=BN_MOMENTUM),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1,
                                                  stride=1, padding=0))

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone(inputs)
        x_line = x[0]

        x_line_laplace = self.line_fuse_layer(F.conv2d(x_line, self.laplace.repeat(32, 32, 1, 1), stride=1, padding=1,))
        x_line_sobel_x = self.line_fuse_layer(F.conv2d(x_line, self.sobel_x.repeat(32, 32, 1, 1), stride=1, padding=1,))
        x_line_sobel_y = self.line_fuse_layer(F.conv2d(x_line, self.sobel_y.repeat(32, 32, 1, 1), stride=1, padding=1,))
        
        x_line = torch.cat([x_line_laplace, x_line_sobel_x, x_line_sobel_y], 1)

        x_line = self.line_last_layer(x_line)

        x_line_pre = F.softmax(x_line, dim=1)
        x_line_pre = torch.argmax(x_line_pre,dim=1)
        x_line_pre = torch.unsqueeze (x_line_pre, 1)


        x0_h, x0_w = x[0].size(2), x[0].size(3)

        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2, x3], 1)
        x = self.scale_atten(x)

        x_glob = self.gcm(x)
        x_local = self.lcf(x)
        x = torch.cat([x_glob, x_local], 1)
        
        x = self.fuse_layer(x)
        x = x+x_line_pre
        x = self.seg_last_layer(x)
        x_line = F.interpolate(x_line, size=(H, W), mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)




        return x, x_line
