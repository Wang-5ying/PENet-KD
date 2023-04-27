from collections import OrderedDict
from math import sqrt

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# from mmseg.models.backbones.mix_transformer import mit_b2
# from xiugai3.uniformer import uniformer_small
from backbone.Shunted.SSA import shunted_t
from xiugai3.api import RRAA, End, BasicConv2d, de, edge, AGG2, Gru
from xiugai3.allkindattention import Channel_aware_CoordAtt
import time

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
    def load_pret(self, pre_model1):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model1)
        for k, v in state_dict.items():
            name = k #[9:]
            new_state_dict3[name] = v
        self.resnet.load_state_dict(new_state_dict3, strict=False)
        self.resnet_depth.load_state_dict(new_state_dict3, strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model1}")
        print(f"Depth SwinTransformer loading pre_model ${pre_model1}")

    def __init__(self):
        super(SRAA, self).__init__()
        # 64 128 320 512
        self.resnet = shunted_t()
        self.resnet_depth = shunted_t()
        self.raa1 = RRAA(in_channel=64, out_channel=64, h=80, w=80)
        self.raa2 = RRAA(in_channel=128, out_channel=128, h=40, w=40)
        self.raa3 = RRAA(in_channel=256, out_channel=320, h=20, w=20)
        self.raa4 = RRAA(in_channel=512, out_channel=512, h=10, w=10)

        self.agg1 = AGG2(32, 96)
        self.agg2 = AGG2(32, 64)
        self.agg3 = AGG2(32, 64)
        self.agg4 = AGG2(32, 64)

        # self.glo = GlobalInfo(in_channel=2048, out_channel2=64)
        self.glo = nn.Conv2d(512, 16, 1)
        self.end1 = End(in_channel=16)

        self.sup1 = nn.Conv2d(512, 1, 3, 1, 1)
        self.sup2 = nn.Conv2d(256, 1, 3, 1, 1)
        self.sup3 = nn.Conv2d(128, 1, 3, 1, 1)

        self.b1 = BasicConv2d(64, 16, 3, 1, 1)
        self.b2 = BasicConv2d(16, 16, 3, 1, 1)
        self.b3 = BasicConv2d(16, 16, 3, 1, 1)
        self.b4 = BasicConv2d(16, 16, 3, 1, 1)
        self.b5 = BasicConv2d(128, 16, 3, 1, 1)

        # self.d1 = de(1536, 512)
        # self.d2 = de(1792, 256)
        # self.d3 = de(1920, 128)

        self.edge1 = edge(64, 1)
        self.edge2 = edge(256, 1)

        self.before1 = BasicConv2d(512, 256, 3, 1, 1)
        self.r1 = BasicConv2d(256, 128, 3, 1, 1)
        self.before2 = BasicConv2d(256, 128, 3, 1, 1)
        self.r2 = BasicConv2d(128, 64, 3, 1, 1)
        self.before3 = BasicConv2d(128, 64, 3, 1, 1)
        self.r3 = BasicConv2d(64, 3, 3, 1, 1)

        self.before1t = BasicConv2d(512, 256, 3, 1, 1)
        self.t1 = BasicConv2d(256, 128, 3, 1, 1)
        self.before2t = BasicConv2d(256, 128, 3, 1, 1)
        self.t2 = BasicConv2d(128, 64, 3, 1, 1)
        self.before3t = BasicConv2d(128, 64, 3, 1, 1)
        self.t3 = BasicConv2d(64, 3, 3, 1, 1)

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
        self.fenliang1 = BasicConv2d(512, 16, 1)
        self.fenliang2 = BasicConv2d(512 + 256, 16, 1)
        self.fenliang3 = BasicConv2d(512 + 256 + 128, 16, 1)
        self.fenliang4 = BasicConv2d(512 + 256 + 128 + 64, 16, 1)

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

        self.cca1 = Channel_aware_CoordAtt(16, 16, 20, 20)
        self.cca2 = Channel_aware_CoordAtt(16, 16, 40, 40)
        self.cca3 = Channel_aware_CoordAtt(16, 16, 80, 80)
        self.cca4 = Channel_aware_CoordAtt(16, 16, 160, 160)

        self.xiao1 = nn.Conv2d(64, 512, 1)
        self.xiao2 = nn.Conv2d(512, 256, 1)
        self.xiao3 = nn.Conv2d(256, 128, 1)
        self.xiao4 = nn.Conv2d(128, 128, 1)
    def forward(self, r, d):
        d = torch.cat((d,d,d), dim=1) + r
        B = r.shape[0]
        # -------->
        # 1
        rlayer_features = self.resnet.forward(r)
        tlayer_features = self.resnet_depth.forward(d)
        # for i in rlayer_features:
        #     print(i.size())
        # gl
        glx = rlayer_features[-1]
        gly = tlayer_features[-1]

        # <----------------RGB
        # r4 = self.before1(F.interpolate(rlayer_features[3], 20, mode='bilinear', align_corners=True))
        # r3 = self.r1(r4 + rlayer_features[2])
        # r3 = F.interpolate(r3, 40, mode='bilinear', align_corners=True)
        # # r3 = self.resnet.layers[1](r3.flatten(2).transpose(1, 2))
        # patch_embed = getattr(self.resnet, f"patch_embed{3}")
        # block = getattr(self.resnet, f"block{3}")
        # norm = getattr(self.resnet, f"norm{3}")
        # x, H, W = patch_embed(r3)
        # for blk in block:
        #     x = blk(x, H, W)
        # x = norm(x)
        # r3 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 64 * 64 *64
        #
        # # B, L, C = r3.shape
        # # r3 = r3.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        # raar3 = r3
        # r3 = self.before2(F.interpolate(r3, 40, mode='bilinear', align_corners=True))
        # r2 = self.r2(r3 + rlayer_features[1])
        # r2 = F.interpolate(r2, 80, mode='bilinear', align_corners=True)
        # # r2 = self.resnet.pos_drop(r2)
        # # r2 = self.resnet.layers[0](r2.flatten(2).transpose(1, 2))
        # patch_embed = getattr(self.resnet, f"patch_embed{2}")
        # block = getattr(self.resnet, f"block{2}")
        # norm = getattr(self.resnet, f"norm{2}")
        # x, H, W = patch_embed(r2)
        # for blk in block:
        #     x = blk(x, H, W)
        # x = norm(x)
        # r2 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #
        # # B, L, C = r2.shape
        # # r2 = r2.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        # raar2 = r2
        # r1 = self.before3(F.interpolate(r2, 80, mode='bilinear', align_corners=True))
        # r1 = self.r3(r1 + rlayer_features[0])
        # r1 = F.interpolate(r1, 320, mode='bilinear', align_corners=True)
        # # r1 = self.resnet.patch_embed(r1)
        # patch_embed = getattr(self.resnet, f"patch_embed{1}")
        # block = getattr(self.resnet, f"block{1}")
        # norm = getattr(self.resnet, f"norm{1}")
        # x, H, W = patch_embed(r1)
        # for blk in block:
        #     x = blk(x, H, W)
        # x = norm(x)
        # r1 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #
        # # B, L, C = r1.shape
        # # r1 = r1.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        # raar1 = r1
        #
        # # <-------------theramal
        # t4 = self.before1t(F.interpolate(tlayer_features[3], 20, mode='bilinear', align_corners=True))
        # t3 = self.t1(t4 + tlayer_features[2])
        # t3 = F.interpolate(t3, 40, mode='bilinear', align_corners=True)
        # # t3 = self.resnet_depth.layers[1](t3.flatten(2).transpose(1, 2))
        # patch_embed = getattr(self.resnet, f"patch_embed{3}")
        # block = getattr(self.resnet, f"block{3}")
        # norm = getattr(self.resnet, f"norm{3}")
        # x, H, W = patch_embed(t3)
        # for blk in block:
        #     x = blk(x, H, W)
        # x = norm(x)
        # t3 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 64 * 64 *64
        #
        # # B, L, C = t3.shape
        # # t3 = t3.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        # raat3 = t3
        # t3 = self.before2t(F.interpolate(t3, 40, mode='bilinear', align_corners=True))
        # t2 = self.t2(t3 + tlayer_features[1])
        # t2 = F.interpolate(t2, 80, mode='bilinear', align_corners=True)
        # # t2 = self.resnet_depth.pos_drop(t2)
        # # t2 = self.resnet_depth.layers[0](t2.flatten(2).transpose(1, 2))
        # patch_embed = getattr(self.resnet, f"patch_embed{2}")
        # block = getattr(self.resnet, f"block{2}")
        # norm = getattr(self.resnet, f"norm{2}")
        # x, H, W = patch_embed(t2)
        # for blk in block:
        #     x = blk(x, H, W)
        # x = norm(x)
        # t2 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #
        # # B, L, C = t2.shape
        # # t2 = t2.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        # raat2 = t2
        # t1 = self.before3t(F.interpolate(t2, 80, mode='bilinear', align_corners=True))
        # t1 = self.t3(t1 + tlayer_features[0])
        # t1 = F.interpolate(t1, 320, mode='bilinear', align_corners=True)
        # # t1 = self.resnet_depth.patch_embed(t1)
        # patch_embed = getattr(self.resnet, f"patch_embed{1}")
        # block = getattr(self.resnet, f"block{1}")
        # norm = getattr(self.resnet, f"norm{1}")
        # x, H, W = patch_embed(t1)
        # for blk in block:
        #     x = blk(x, H, W)
        # x = norm(x)
        # t1 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #
        # # B, L, C = t1.shape
        # # t1 = t1.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        # raat1 = t1

        # rd1 = self.raa1(raar1, raat1)
        # rd2 = self.raa2(raar2, raat2)
        # rd3 = self.raa3(raar3, raat3)
        # rd4 = self.raa4(rlayer_features[3], tlayer_features[3])

        rd1 = self.raa1(rlayer_features[0], tlayer_features[0])
        rd2 = self.raa2(rlayer_features[1], tlayer_features[1])
        rd3 = self.raa3(rlayer_features[2], tlayer_features[2])
        rd4 = self.raa4(rlayer_features[3], tlayer_features[3])
        # rd1 = rlayer_features[0] * tlayer_features[0]
        # rd2 = rlayer_features[0] * tlayer_features[0]
        # rd3 = rlayer_features[0] * tlayer_features[0]
        # rd4 = rlayer_features[3] * tlayer_features[3]

        decode = []
        # decoder
        # B, L, C = glx.shape
        # glx = glx.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        # B, L, C = gly.shape
        # gly = gly.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        glo = F.interpolate(self.glo(glx + gly), 20)

        # b1 = self.b1(rd1)
        # b1 = F.interpolate(b1, 20, mode='bilinear', align_corners=True)
        # a1 = F.interpolate(rd4, 20, mode='bilinear', align_corners=True)
        # a1 = self.fenliang1(a1)
        # a1cw = self.cw1(self.beforecw1(a1), self.beforecw1(a1))
        # a1cw1 = self.aftercw1(F.interpolate(a1cw, size=20))
        # res = self.agg1(b1 * a1cw1, a1, self.cca1(glo))
        res1 = F.interpolate(self.xiao1(rd1), size=20) * F.interpolate(rd4, size=20)
        decode.append(res1)

        res2 = F.interpolate(self.xiao2(res1), size=40) * F.interpolate(rd3, size=40)
        decode.append(res2)

        res3 = F.interpolate(self.xiao3(res2), size=80) * F.interpolate(rd2, size=80)
        decode.append(res3)

        res4 = F.interpolate(self.xiao4(res3), size=160) * F.interpolate(rd2, size=160)
        decode.append(res4)

        res = self.b5(res3)
        res = self.end1(res)
        # supervision
        b2 = self.sup1(res1)
        b3 = self.sup2(res2)
        b4 = self.sup3(res3)
        b2 = F.interpolate(b2, 320)
        b3 = F.interpolate(b3, 320)
        b4 = F.interpolate(b4, 320)

        edge1 = self.edge1(rd1)
        edge2 = self.edge2(rd3)
        edge1 = F.interpolate(edge1, 320)
        edge2 = F.interpolate(edge2, 320)

        # return res, b2, b3, b4, edge1, edge2
        return res, res

if __name__ == '__main__':
    a = torch.randn(1, 3, 384, 384).cuda()
    b = torch.randn(1, 1, 384, 384).cuda()
    model = SRAA()
    model.cuda()
    # model.load_pret("/home/wby/uniformer/uniformer_base_tl_384.pth")
    out = model(a, b)
    # end = time.time()
    with torch.no_grad():
        start = time.time()
        out = model(a, b)
        end = time.time()
    print(1/(end-start))
