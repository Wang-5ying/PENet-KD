from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F

class CA(nn.Module):
    def __init__(self,in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()
    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)
class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class AGG2(nn.Module):
    def __init__(self, in_channel1,in_channel2):
        super(AGG2, self).__init__()
        self.branch1 = BasicConv2d(in_channel1, in_channel1, kernel_size=3, padding=1)
        self.ca=CA(in_ch=in_channel1)
    def forward(self,x1,x2,glo):
        f_sum = torch.cat((x1, x2), 1)
        f_sum=self.ca(self.branch1(f_sum))
        weight = F.softmax(f_sum, dim=1)
        w2, w1 = torch.chunk(weight, 2, 1)
        # print(w1.size(),w2.size(),x1.size(),x2.size())
        prediction = w2 * x1 + w1 * x2
        prediction = prediction*glo
        return prediction



class ARAA(nn.Module):
    def __init__(self,in_channel,out_channel,h,w):
        super(ARAA,self).__init__()
        self.ca1 = CA(in_channel)
        self.sa=SA()
        self.conv1=BasicConv2d(in_planes=in_channel*2,out_planes=out_channel,kernel_size=3,stride=1,padding=1)
        self.conv2 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=7, stride=1, padding=3)
        self.ra=RRA(in_channel,out_channel)
    def forward(self,rx,dx):
        d1 = self.ca1(dx)
        d2=d1.mul(dx)
        res=torch.cat((d2,rx),dim=1)
        res=self.conv1(res)
        res=self.conv2(res)
        out=self.ra(rx,res)
        return out

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        return h

class Gru(nn.Module):

    def __init__(self, num_in, num_mid, h,stride=(1,1), kernel=1):
        super(Gru, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)
        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)
        # reduce dimension
        self.conv_state = BasicConv2d(num_in, self.num_s,  kernel_size=kernel_size, padding=padding)
        # generate projection and inverse projection functions
        self.conv_proj = BasicConv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)

        self.conv_state2 = BasicConv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate projection and inverse projection functions
        self.conv_proj2 = BasicConv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)

        # reasoning by graph convolution
        self.gcn1 = GCN(num_state=self.num_s, num_node=self.num_n)
        self.gcn2 = GCN(num_state=self.num_s, num_node=self.num_n)
        # fusion
        self.fc_2 = nn.Conv2d(num_in, num_in, kernel_size=kernel_size, padding=padding, stride=(1,1),
                              groups=1, bias=False)
        self.blocker = nn.BatchNorm2d(num_in)
        self.h = h
    def forward(self, x, y):
        # print("x",x.size())
        batch_size = x.size(0)

        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1)
        y_proj_reshaped = self.conv_proj(y).view(batch_size, self.num_n, -1)

        x_state_2 = self.conv_state2(x).view(batch_size, self.num_s, -1)


        x_n_state1 = torch.bmm(x_state_reshaped, y_proj_reshaped.permute(0, 2, 1))
        x_n_state2 = x_n_state1 * (1. / x_state_reshaped.size(2))



        x_n_rel1 = self.gcn1(x_n_state2)
        # print("x_n_real1",x_n_rel1.size())
        x_n_rel2 = self.gcn2(x_n_rel1)
        # print("x_n_real2", x_n_rel2.size())
        # inverse project to original space
        # print(x_n_rel2.size(), x_state_2.size())
        x_state_reshaped = torch.bmm(x_n_rel2.permute(0,2,1), x_state_2)
        # print(x_state_reshaped.size())
        B, C, N = x_state_reshaped.shape
        x_state = x_state_reshaped.view(batch_size, 1, int(sqrt(N)), int(sqrt(N)))
        # print(x_state.size())
        # fusion
        out = x + self.blocker(self.fc_2(x_state)) + y
        # print(out.size())

        return out
class RRAA(nn.Module):
    def __init__(self, in_channel, out_channel, h, w):
            super(RRAA, self).__init__()
            self.ca1 = CA(in_channel)
            self.ca2 = CA(in_channel)
            self.sa1=SA()
            self.conv1=nn.Sequential(nn.Conv2d(in_channels=in_channel*2, out_channels=1,kernel_size=3,stride=1,padding=1),
                                     nn.BatchNorm2d(1),
                                     nn.ReLU())
            self.conv2 =nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=1,kernel_size=3,padding= 1, dilation=1),
                                     nn.BatchNorm2d(1),
                                     nn.ReLU())
            self.conv3 =nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=1,kernel_size=3,padding= 1, dilation=1),
                                     nn.BatchNorm2d(1),
                                     nn.ReLU())
            self.conv4 =nn.Sequential(nn.Conv2d(in_channels=in_channel+1, out_channels=in_channel,kernel_size=3,padding= 1, dilation=1),
                                     nn.BatchNorm2d(in_channel),
                                     nn.ReLU())
            self.ghl1 = Gru(1,1,h)
            self.ghl2 = Gru(1,1,h)
    def forward(self, r, d):
            rx=self.ca1(r)
            dx=self.ca2(d)
            rdx=torch.cat((rx,dx),dim=1)
            rdx=self.conv1(rdx)

            rxg=self.conv2(rx)
            dxg=self.conv3(dx)

            # print("rxg",rxg.size(),dxg.size())
            rxg = self.ghl1(rxg, dxg)
            dxg = self.ghl2(dxg, rxg)

            c1=rdx.mul(rxg)
            c2=rdx.mul(dxg)

            c1=torch.cat((c1,rx),dim=1)
            c2=torch.cat((c2,dx),dim=1)
            # c1 = c1.mul(c1g)
            # c2 = c2.mul(c2g)
            res=c1+c2
            res=self.conv4(res)
            res=res+d+r
            return res

class RRA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RRA, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        #27
        self.convert2 = nn.Conv2d(in_channel, in_channel, 1)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 3, padding=1),
        )
        self.channel = out_channel
        self.trans=nn.Conv2d(out_channel,1,1)
    def forward(self, x, y):
        a=self.convert2(y)
        x = self.convert(x)
        x = a.expand(-1, self.channel, -1, -1).mul(x)
        y = y + self.convs(x)
        return  y

class End(nn.Module):
    def __init__(self,in_channel):
        super(End,self).__init__()
        self.r55 = nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=5, padding=2)
        # self.bn55 = nn.BatchNorm2d(1)
        # self.re55 = nn.ReLU()
    def forward(self,x):
        res1 = torch.nn.functional.upsample(x, (320,320))
        res1 = self.r55(res1)
        # res1 = self.bn55(res1)
        # res1 = self.re55(res1)
        return res1

def convblock(in_,out_,ks,st,pad):
    return nn.Sequential(
        nn.Conv2d(in_,out_,ks,st,pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )
class CA(nn.Module):
    def __init__(self,in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()
    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)
class GlobalInfo(nn.Module):
    def __init__(self,in_channel,out_channel2):
        super(GlobalInfo, self).__init__()
        self.ca = CA(in_channel)
        self.de_chan = convblock(in_channel, out_channel2, 3, 1, 1)

        self.b0 = nn.Sequential(
            nn.AdaptiveMaxPool2d(13),
            nn.Conv2d(out_channel2, out_channel2, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.b1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(9),
            nn.Conv2d(out_channel2, out_channel2, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.AdaptiveMaxPool2d(7),
            nn.Conv2d(out_channel2, out_channel2, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.AdaptiveMaxPool2d(5),
            nn.Conv2d(out_channel2, out_channel2, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.AdaptiveMaxPool2d(3),
            nn.Conv2d(out_channel2, out_channel2, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fus = convblock(out_channel2*6, out_channel2, 1, 1, 0)

    def forward(self, rgb, d):
        x_size = rgb.size()[2:]
        x = self.ca(torch.cat((rgb, d), 1))
        x = self.de_chan(x)
        b0 = F.interpolate(self.b0(x), x_size, mode='bilinear', align_corners=True)
        b1 = F.interpolate(self.b1(x), x_size, mode='bilinear', align_corners=True)
        b2 = F.interpolate(self.b2(x), x_size, mode='bilinear', align_corners=True)
        b3 = F.interpolate(self.b3(x), x_size, mode='bilinear', align_corners=True)
        b4 = F.interpolate(self.b3(x), x_size, mode='bilinear', align_corners=True)
        out = self.fus(torch.cat((b0, b1,b2,b3,b4, x), 1))
        return out

class de(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(de,self).__init__()
        self.conv1=nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=1)
        self.conv2=nn.Conv2d(in_channels=outchannel,out_channels=outchannel,kernel_size=3,padding=1)
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        return x

class edge(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(edge,self).__init__()
        self.conv1=nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=1)
        self.conv2=nn.Conv2d(in_channels=outchannel,out_channels=outchannel,kernel_size=3,padding=1)
    def forward(self,x1):
        x = self.conv1(x1)
        x = self.conv2(x)
        return x