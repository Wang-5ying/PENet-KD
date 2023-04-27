from math import sqrt

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from First_model.AMajorchanges.PENet.backbone.SwinNet.models.Swin_Transformer import SwinTransformer
from xiugai3.api import RRAA, End, BasicConv2d, de, edge, AGG2, Gru
from First_model.AMajorchanges.xiugai3.allkindattention import Channel_aware_CoordAtt


class CAttention(nn.Module):
    def __init__(self, dim, reduction=8, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv_r = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_f = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, rgb, fuse):
        B, C, H, W = rgb.shape
        rgb = rgb.reshape(B, H * W, C)
        B, N, C = rgb.shape
        B, C, H, W = fuse.shape
        fuse = fuse.reshape(B, H * W, C)
        B, N, C = fuse.shape
        print("1", rgb.size(), fuse.size())
        qkv_r = self.qkv_r(rgb).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qr, kr, vr = qkv_r[0], qkv_r[1], qkv_r[2]  # make torchscript happy (cannot use tensor as tuple)
        qkv_f = self.qkv_f(fuse).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qf, kf, vf = qkv_f[0], qkv_f[1], qkv_f[2]  # make torchscript happy (cannot use tensor as tuple)
        attn_r = (qf @ kr.transpose(-2, -1)) * self.scale
        # attn_r = (qf @ kr) * self.scale
        attn_r = attn_r.softmax(dim=-1)
        attn_r = self.attn_drop(attn_r)
        rgb_a = (attn_r @ vr).transpose(1, 2).reshape(B, N, C)
        rgb_a = self.proj(rgb_a)
        rgb_a = self.proj_drop(rgb_a)

        attn_f = (qr @ kf.transpose(-2, -1)) * self.scale
        attn_f = attn_f.softmax(dim=-1)
        attn_f = self.attn_drop(attn_f)
        fuse_a = (attn_f @ vf).transpose(1, 2).reshape(B, N, C)
        fuse_a = self.proj(fuse_a) + fuse
        fuse_a = self.proj_drop(fuse_a)

        B, N, C = rgb_a.shape
        rgb_a = rgb_a.reshape(B, C, int(sqrt(N)), int(sqrt(N)))
        print(fuse_a.size(), rgb_a.size())
        return rgb_a, fuse_a


class SRAA(nn.Module):
    def load_pret(self, pre_model):
        self.resnet.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.resnet_depth.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")

    def __init__(self):
        super(SRAA, self).__init__()
        self.resnet = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.resnet_depth = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.raa1 = RRAA(in_channel=128, out_channel=128, h=96, w=96)
        self.raa2 = RRAA(in_channel=256, out_channel=256, h=48, w=48)
        self.raa3 = RRAA(in_channel=512, out_channel=512, h=24, w=24)
        self.raa4 = RRAA(in_channel=1024, out_channel=1024, h=12, w=12)

        self.agg1 = AGG2(32, 48)
        self.agg2 = AGG2(32, 64)
        self.agg3 = AGG2(32, 64)
        self.agg4 = AGG2(32, 64)

        # self.glo = GlobalInfo(in_channel=2048, out_channel2=64)
        self.glo = nn.Conv2d(1024, 16, 1)
        self.end1 = End(in_channel=16)

        self.sup1 = BasicConv2d(16, 1, 3, 1, 1)
        self.sup2 = BasicConv2d(16, 1, 3, 1, 1)
        self.sup3 = BasicConv2d(16, 1, 3, 1, 1)

        self.b1 = BasicConv2d(128, 16, 3, 1, 1)
        self.b2 = BasicConv2d(16, 16, 3, 1, 1)
        self.b3 = BasicConv2d(16, 16, 3, 1, 1)
        self.b4 = BasicConv2d(16, 16, 3, 1, 1)
        self.b5 = BasicConv2d(16, 16, 3, 1, 1)

        # self.d1 = de(1536, 512)
        # self.d2 = de(1792, 256)
        # self.d3 = de(1920, 128)

        self.edge1 = edge(128, 1)
        self.edge2 = edge(512, 1)

        self.before1 = BasicConv2d(1024, 512, 3, 1, 1)
        self.r1 = BasicConv2d(512, 256, 3, 1, 1)
        self.before2 = BasicConv2d(512, 256, 3, 1, 1)
        self.r2 = BasicConv2d(256, 128, 3, 1, 1)
        self.before3 = BasicConv2d(256, 128, 3, 1, 1)
        self.r3 = BasicConv2d(128, 3, 3, 1, 1)

        self.before1t = BasicConv2d(1024, 512, 3, 1, 1)
        self.t1 = BasicConv2d(512, 256, 3, 1, 1)
        self.before2t = BasicConv2d(512, 256, 3, 1, 1)
        self.t2 = BasicConv2d(256, 128, 3, 1, 1)
        self.before3t = BasicConv2d(256, 128, 3, 1, 1)
        self.t3 = BasicConv2d(128, 3, 3, 1, 1)

        # self.xiaoagg1=nn.Conv2d(1024,64,1)
        # self.xiaoagg2rd34=nn.Conv2d(1536,64,1)
        # self.xiaoagg3rd234 = nn.Conv2d(1792, 64, 1)
        # self.xiaoagg4rd1234=nn.Conv2d(1920,64,1)
        # self.xiaoraa1=nn.Conv2d(256,128,1)
        # self.xiaoraa2 = nn.Conv2d(512, 256, 1)
        # self.xiaoraa3 = nn.Conv2d(1024, 512, 1)
        # self.xiaoraa4 = nn.Conv2d(2048, 1024, 1)

        # self.xcagg1=nn.Conv2d(64+64+1024,64,1)
        # self.xcagg2 = nn.Conv2d(64+1024+512+64, 64, 1)
        # self.xcagg3 = nn.Conv2d(64 + 1024 + 512 +256+ 64, 64, 1)
        # self.xcagg4 = nn.Conv2d(64 + 1024 + 512 + 256 + 128 + 64, 64, 1)

        # self.u1=nn.Conv2d(1024,512,1)
        # self.u2 = nn.Conv2d(512, 256, 1)
        # self.u3 = nn.Conv2d(256, 128, 1)
        # self.u4 = nn.Conv2d(128, 1, 1)
        self.fenliang1 = BasicConv2d(1024, 16, 1)
        self.fenliang2 = BasicConv2d(1024 + 512, 16, 1)
        self.fenliang3 = BasicConv2d(1024 + 512 + 256, 16, 1)
        self.fenliang4 = BasicConv2d(1024 + 512 + 256 + 128, 16, 1)

        self.glc1 = BasicConv2d(16, 64, 3, 1, 1)
        self.glc2 = BasicConv2d(16, 256, 3, 1, 1)
        self.glc3 = BasicConv2d(16, 1024, 3, 1, 1)

        self.cw1 = Gru(1, 1, 12)
        self.cw2 = Gru(1, 1, 24)
        self.cw3 = Gru(1, 1, 48)
        self.cw4 = Gru(1, 1, 96)

        self.beforecw1 = nn.Conv2d(16, 1, 1)
        self.beforecw2 = nn.Conv2d(16, 1, 1)
        self.beforecw3 = nn.Conv2d(16, 1, 1)
        self.beforecw4 = nn.Conv2d(16, 1, 1)

        self.aftercw1 = nn.Conv2d(1, 1, 1)
        self.aftercw2 = nn.Conv2d(1, 1, 1)
        self.aftercw3 = nn.Conv2d(1, 1, 1)
        self.aftercw4 = nn.Conv2d(1, 1, 1)

        self.cca1 = Channel_aware_CoordAtt(16, 16, 12, 12)
        self.cca2 = Channel_aware_CoordAtt(16, 16, 24, 24)
        self.cca3 = Channel_aware_CoordAtt(16, 16, 48, 48)
        self.cca4 = Channel_aware_CoordAtt(16, 16, 96, 96)

    def forward(self, r, d):
        # -------->
        # 1
        rlayer_features = []
        x1 = self.resnet.patch_embed(r)
        B, L, C = x1.shape
        rlayer_features.append(x1.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous())
        x1 = self.resnet.pos_drop(x1)

        tlayer_features = []
        y1 = self.resnet_depth.patch_embed(d)
        B, L, C = y1.shape
        tlayer_features.append(y1.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous())
        y1 = self.resnet_depth.pos_drop(y1)
        # raa
        # rd1 = self.raa1(rlayer_features[0], tlayer_features[0])

        # 2
        x2 = self.resnet.layers[0](x1)
        B, L, C = x2.shape
        xl2 = x2.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        rlayer_features.append(xl2)

        y2 = self.resnet_depth.layers[0](y1)
        B, L, C = y2.shape
        xly2 = y2.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        tlayer_features.append(xly2)

        # 3
        x3 = self.resnet.layers[1](x2)
        B, L, C = x3.shape
        xl3 = x3.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        rlayer_features.append(xl3)

        y3 = self.resnet_depth.layers[1](y2)
        B, L, C = y3.shape
        xly3 = y3.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        tlayer_features.append(xly3)

        # 4
        x4 = self.resnet.layers[2](x3)
        B, L, C = x4.shape
        xl4 = x4.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        rlayer_features.append(xl4)

        y4 = self.resnet_depth.layers[2](y3)
        B, L, C = y4.shape
        xly4 = y4.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        tlayer_features.append(xly4)

        # gl
        glx = x4
        gly = y4

        # <----------------RGB
        r4 = self.before1(F.interpolate(rlayer_features[3], 24, mode='bilinear', align_corners=True))
        r3 = self.r1(r4 + rlayer_features[2])
        r3 = F.interpolate(r3, 48, mode='bilinear', align_corners=True)
        r3 = self.resnet.layers[1](r3.flatten(2).transpose(1, 2))

        B, L, C = r3.shape
        r3 = r3.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        raar3 = r3
        r3 = self.before2(F.interpolate(r3, 48, mode='bilinear', align_corners=True))
        r2 = self.r2(r3 + rlayer_features[1])
        r2 = F.interpolate(r2, 96, mode='bilinear', align_corners=True)
        r2 = self.resnet.pos_drop(r2)
        r2 = self.resnet.layers[0](r2.flatten(2).transpose(1, 2))

        B, L, C = r2.shape
        r2 = r2.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        raar2 = r2
        r1 = self.before3(F.interpolate(r2, 96, mode='bilinear', align_corners=True))
        r1 = self.r3(r1 + rlayer_features[0])
        r1 = F.interpolate(r1, 384, mode='bilinear', align_corners=True)
        r1 = self.resnet.patch_embed(r1)

        B, L, C = r1.shape
        r1 = r1.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        raar1 = r1

        # <-------------theramal
        t4 = self.before1t(F.interpolate(tlayer_features[3], 24, mode='bilinear', align_corners=True))
        t3 = self.t1(t4 + tlayer_features[2])
        t3 = F.interpolate(t3, 48, mode='bilinear', align_corners=True)
        t3 = self.resnet_depth.layers[1](t3.flatten(2).transpose(1, 2))

        B, L, C = t3.shape
        t3 = t3.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        raat3 = t3
        t3 = self.before2t(F.interpolate(t3, 48, mode='bilinear', align_corners=True))
        t2 = self.t2(t3 + tlayer_features[1])
        t2 = F.interpolate(t2, 96, mode='bilinear', align_corners=True)
        t2 = self.resnet_depth.pos_drop(t2)
        t2 = self.resnet_depth.layers[0](t2.flatten(2).transpose(1, 2))

        B, L, C = t2.shape
        t2 = t2.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        raat2 = t2
        t1 = self.before3t(F.interpolate(t2, 96, mode='bilinear', align_corners=True))
        t1 = self.t3(t1 + tlayer_features[0])
        t1 = F.interpolate(t1, 384, mode='bilinear', align_corners=True)
        t1 = self.resnet_depth.patch_embed(t1)

        B, L, C = t1.shape
        t1 = t1.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        raat1 = t1

        rd1 = self.raa1(raar1, raat1)
        rd2 = self.raa2(raar2, raat2)
        rd3 = self.raa3(raar3, raat3)
        rd4 = self.raa4(rlayer_features[3], tlayer_features[3])
        # rd1 = raar1+ raat1
        # rd2 = raar2+ raat2
        # rd3 = raar3+raat3
        # rd4 = rlayer_features[3]+ tlayer_features[3]

        # decoder
        B, L, C = glx.shape
        glx = glx.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        B, L, C = gly.shape
        gly = gly.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        glo = self.glo(glx + gly)

        # print(glo.size(),rd1.size(),rd2.size(),rd3.size(),rd4.size())
        # glo1 = F.interpolate(glo, 12, mode='bilinear', align_corners=True)
        b1 = self.b1(rd1)
        b1 = F.interpolate(b1, 12, mode='bilinear', align_corners=True)
        a1 = F.interpolate(rd4, 12, mode='bilinear', align_corners=True)
        a1 = self.fenliang1(a1)
        a1cw = self.cw1(self.beforecw1(a1), self.beforecw1(a1))
        a1cw1 = self.aftercw1(F.interpolate(a1cw, size=12))
        res = self.agg1(b1 * a1cw1, a1, self.cca1(glo))

        # glo2 = F.interpolate(glo, 28, mode='bilinear', align_corners=True)
        glo2 = self.glc1(glo)
        glo2 = F.pixel_shuffle(glo2, 2)
        glo2 = self.cca2(glo2)
        res1 = self.b2(res)
        res1 = F.interpolate(res1, 24, mode='bilinear', align_corners=True)
        a1 = F.interpolate(rd4, 24, mode='bilinear', align_corners=True)
        a2 = F.interpolate(rd3, 24, mode='bilinear', align_corners=True)
        a12 = self.fenliang2(torch.cat((a1, a2), dim=1))
        a12cw = self.cw1(self.beforecw1(a12), self.beforecw1(a12))
        a12cw2 = self.aftercw1(F.interpolate(a12cw, size=24))
        res1 = self.agg2(res1 * a12cw2, a12, glo2)

        # glo3 = F.interpolate(glo, 48, mode='bilinear', align_corners=True)
        glo3 = self.glc2(glo)
        glo3 = self.cca3(F.pixel_shuffle(glo3, 4))
        res2 = self.b3(res1)
        res2 = F.interpolate(res2, 48, mode='bilinear', align_corners=True)
        a1 = F.interpolate(rd4, 48, mode='bilinear', align_corners=True)
        a2 = F.interpolate(rd3, 48, mode='bilinear', align_corners=True)
        a3 = F.interpolate(rd2, 48, mode='bilinear', align_corners=True)
        a123 = self.fenliang3(torch.cat((a1, a2, a3), dim=1))
        a123cw3 = self.cw1(self.beforecw1(a123), self.beforecw1(a123))
        a123cw3 = self.aftercw1(F.interpolate(a123cw3, size=48))
        res2 = self.agg3(res2 * a123cw3, a123, glo3)

        # glo4 = F.interpolate(glo, 96, mode='bilinear', align_corners=True)
        glo4 = self.glc3(glo)
        glo4 = self.cca4(F.pixel_shuffle(glo4, 8))
        res3 = self.b3(res2)
        res3 = F.interpolate(res3, 96, mode='bilinear', align_corners=True)
        a1 = F.interpolate(rd4, 96, mode='bilinear', align_corners=True)
        a2 = F.interpolate(rd3, 96, mode='bilinear', align_corners=True)
        a3 = F.interpolate(rd2, 96, mode='bilinear', align_corners=True)
        a4 = F.interpolate(rd1, 96, mode='bilinear', align_corners=True)
        a1234 = self.fenliang4(torch.cat((a1, a2, a3, a4), dim=1))
        a1234cw4 = self.cw1(self.beforecw1(a1234), self.beforecw1(a1234))
        a1234cw4 = self.aftercw1(F.interpolate(a1234cw4, size=96))
        # print(res3.size(),a1234cw4.size())
        res3 = self.agg4(res3 * a1234cw4, a1234, glo4)

        res = self.b5(res3)
        res = self.end1(res)
        # supervision
        b2 = self.sup1(res1)
        b3 = self.sup2(res2)
        b4 = self.sup3(res3)
        b2 = F.interpolate(b2, 384)
        b3 = F.interpolate(b3, 384)
        b4 = F.interpolate(b4, 384)

        edge1 = self.edge1(rd1)
        edge2 = self.edge2(rd3)
        edge1 = F.interpolate(edge1, 384)
        edge2 = F.interpolate(edge2, 384)
        if self.training:
            return res, b2, b3, b4, edge1, edge2
        else:
            return b2, b3, b4, res


if __name__ == '__main__':
    a = torch.randn(2, 3, 384, 384).cuda()
    b = torch.randn(2, 3, 384, 384).cuda()
    model = SRAA()
    model.cuda()
    # model.load_pre("/media/wby/shuju/seg_pre/segformer.b2.512x512.ade.160k.pth")
    out = model(a, b)
    # end = time.time()
