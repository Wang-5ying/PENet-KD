import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
from torch import nn
# from RGBT_dataprocessing_CNet import trainData, valData
from RGBT_dataprocessing_CNet import trainData, valData
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
# import Loss.lovasz_losses as lovasz
import pytorch_iou
# from Net.model.ENet_mobilenet.mb4_add1 import net2
# from model.foruth_net_shunted_new import *
# from Yang.new3_decoder import *
import torchvision
import time
import shutil
from log import get_logger
# from  Loss.Binary_Dice_loss import BinaryDiceLoss
# from Loss.Focal_loss import sigmoid_focal_loss
from xiugai3.student_uniformer import SRAA as teacher
from xiugai3.student_uniformer_t_MOBVIT import SRAA as student
import matplotlib.pyplot as plt

from demo import KLDLoss
def print_network(model,name):
    num_params = 0
    for p in model.parameters():
        num_params +=p.numel()
    print(name)
    print("The number of parameters:{}M".format(num_params/1000000))


IOU = pytorch_iou.IOU(size_average = True).cuda()

class BCELOSS(nn.Module):
    def __init__(self):
        super(BCELOSS, self).__init__()
        self.nll_lose = nn.BCELoss()

    def forward(self, input_scale, taeget_scale):
        losses = []
        for inputs, targets in zip(input_scale, taeget_scale):
            lossall = self.nll_lose(inputs, targets)
            losses.append(lossall)
        total_loss = sum(losses)
        return total_loss

################################################################################################################
batchsize = 4
################################################################################################################

train_dataloader = DataLoader(trainData, batch_size=batchsize, shuffle=True, num_workers=4, drop_last=True)

test_dataloader = DataLoader(valData,batch_size=batchsize,shuffle=True,num_workers=4)


# net = fourmodel_student()
net1 = teacher()
net2 = student()
net1.load_state_dict(torch.load('/home/user/Documents/wby/Pth/549/wanzhen_rail_2023_04_01_08_34_best.pth'),strict=False)
# net1.load_pret("/home/user/Documents/wby/Pth/teacher_best/wanzhen_rail_2023_04_01_08_34_best.pth")
net2.load_pret("/home/user/Documents/wby/backbone/mobilevit/xxsmodel_best.pth.tar")
# net2.load_pret("/home/user/Documents/wby/Pth/teacher_best/wanzhen_rail_2023_04_01_08_34_best.pth")
# net.load_pret('/media/user/shuju/uniformer_base_tl_384.pth')
# net.load_state_dict(torch.load('/home/hjk/文档/third_model_GCN/backbone/Shunted/ckpt_T.pth'))
net1 = net1.cuda()
net2 = net2.cuda()
################################################################################################################
model = 'wanzhen_rail_' + time.strftime("%Y_%m_%d_%H_%M")
print_network(net2, model)
################################################################################################################
bestpath = './Pth/' + model + '_best.pth'
lastpath = './Pth/' + model + '_last.pth'
################################################################################################################
criterion1 = BCELOSS().cuda()
criterion2 = BCELOSS().cuda()
criterion3 = BCELOSS().cuda()
criterion4 = BCELOSS().cuda()
criterion5 = BCELOSS().cuda()
criterion6 = BCELOSS().cuda()
criterion7 = BCELOSS().cuda()
criterion8 = BCELOSS().cuda()

criterion = torch.nn.BCEWithLogitsLoss().cuda()
# focaloss = sigmoid_focal_loss().cuda()
# diceloss = BinaryDiceLoss().cuda()

criterion_val = BCELOSS().cuda()
kld = KLDLoss().cuda()
def dice_loss(pred, mask):
    intersection = (pred * mask).sum(axis=(2, 3))
    unior = (pred + mask).sum(axis=(2, 3))
    dice = (2 * intersection + 1) / (unior + 1)
    dice = torch.mean(1 - dice)
    return dice

def kd_loss(pred, mask):
    b1, c1, h1 ,w1 = pred.shape
    b2, c2, h2, w2 = mask.shape
    pred_reshape = pred.reshape(b1, c1, -1)
    pred_tr = pred_reshape.permute(0, 2, 1)
    mask_reshape = mask.reshape(b2, c2, -1)
    mask_tr = mask_reshape.permute(0, 2, 1)
    mul_pred = torch.bmm(pred_tr, pred_reshape)
    mul_mask = torch.bmm(mask_tr, mask_reshape)
    softmax_pred = F.softmax(mul_pred/np.sqrt(c1), dim=0)
    logsoftmax = nn.LogSoftmax(dim=0)
    softmax_mask = logsoftmax(mul_mask/np.sqrt(c2))
    loss = (torch.sum(- softmax_pred * softmax_mask))/w1/h1
    return loss


def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n,c,h,w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4,2,1]:
            if l >=h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
            tmpft = F.adaptive_avg_pool2d(ft, (l,l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all

################################################################################################################
lr_rate = 1e-4 # 1e-4
optimizer = optim.Adam(net2.parameters(), lr=lr_rate)  # , weight_decay=1e-3
################################################################################################################

best = [10]
step = 0
mae_sum = 0
best_mae = 1
best_epoch = 0

logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}({model})'
if not os.path.exists(logdir):
    os.makedirs(logdir)

logger = get_logger(logdir)
logger.info(f'Conf | use logdir {logdir}')

################################################################################################################
epochs = 120
################################################################################################################
temp = 34
logger.info(f'Epochs:{epochs}  Batchsize:{batchsize}')
for epoch in range(epochs):
    mae_sum = 0
    trainmae = 0
    if (epoch+1) % 20 == 0 and epoch != 0:
        for group in optimizer.param_groups:
            group['lr'] = 0.5 * group['lr']
            print(group['lr'])
            lr_rate = group['lr']


    train_loss = 0
    net2 = net2.train()
    prec_time = datetime.now()
    if temp > 3:
        temp = temp -3
    for i, sample in enumerate(train_dataloader):

        image = Variable(sample['RGB'].cuda())
        # image = Variable(sample['RGB'].float().cuda())
        depth = Variable(sample['depth'].cuda())
        label = Variable(sample['label'].float().cuda())
        bound = Variable(sample['bound'].float().cuda())
        background1 = 1 - label

        optimizer.zero_grad()

        # out1, out2, out3, out4, out5 = net(image, depth)
        with torch.no_grad():
            s4_out_t, s3_out_t, s2_out_t, s1_out_t, out_r_t, out_d_t, rlayer_features_t, tlayer_features_t, decode_t = net1(image, depth)
        s4_out, s3_out, s2_out, s1_out, out_r, out_d, rlayer_features, tlayer_features, decode, adj = net2(image, depth)
        # out1, out2, out3, out4, r1, d1, r4, d4, s1_new, Z4, S4, rd4 = net(image, depth)

        # s4_out = F.sigmoid(s4_out)
        # s3_out = F.sigmoid(s3_out)
        # s2_out = F.sigmoid(s2_out)
        # s1_out = F.sigmoid(s1_out)
        # out_r = F.sigmoid(out_r)
        # out_d = F.sigmoid(out_d)
        # o1 = F.sigmoid(out1)
        # o2 = F.sigmoid(out2)
        # o3 = F.sigmoid(out3)
        # o4 = F.sigmoid(out4)
        # o5 = F.sigmoid(out5)


        # bs4_out = 1 - s4_out
        # bs3_out = 1 - s3_out
        # bs2_out = 1 - s2_out
        # bs1_out = 1 - s1_out
        # bout_r = 1 - out_r
        # bout_d = 1 - out_d
        # b_out_new_r = 1 - out_new_r
        #
        fore_loss1 = criterion(s3_out, label) #+ IOU(s3_out, label)
        fore_loss2 = criterion(s2_out, label) #+ IOU(s2_out, label)
        fore_loss3 = criterion(s1_out, label) #+ IOU(s1_out, label)
        # fore_loss4 = criterion1(out_r, label) #+ IOU(out_r, label)
        # fore_loss5 = criterion1(out_d, bound) + IOU(out_d, bound)
        fore_loss6 = criterion(s4_out, label) #+ IOU(s4_out, bound)
        # fore_loss1 = criterion1(o1, label) + IOU(o1, label)
        # fore_loss2 = criterion1(o2, label) + IOU(o2, label)
        # fore_loss3= criterion1(o3, label) + IOU(o3, label)
        # fore_loss4 = criterion1(o4, label) + IOU(o4, label)
        # fore_loss5 = criterion1(o5, label) + IOU(o5, label)
        self = fore_loss1 + fore_loss2 + fore_loss3 + fore_loss6

        # back_loss1 = (criterion1(bs4_out, background1) + IOU(bs4_out, background1))
        # back_loss2 = (criterion1(bs3_out, background1) + IOU(bs3_out, background1))
        # back_loss3 = (criterion1(bs2_out, background1) + IOU(bs2_out, background1))
        # back_loss4 = (criterion1(bs1_out, background1) + IOU(bs1_out, background1))
        # back_loss5 = (criterion1(bout_r, background1) + IOU(bout_r, background1))
        # back_loss6 = (criterion1(bout_d, background1) + IOU(bout_d, background1))

        # -----distill-------------
        # kd1 = kd_loss(s4_out, s4_out_t)
        # print(s4_out.size(), s4_out_t.size())    # s4_out, s3_out, s2_out, s1_out, out_r, out_d
        kd1 = (1-0.0625) * kld(s4_out/4, s4_out_t/4, s4_out, 3) + 0.0625 * criterion(s4_out, label)
        kd2 = (1-0.0625) * kld(s3_out/4, s3_out_t/4, s3_out, 3) + 0.0625 * criterion(s3_out, label)
        kd3 = (1-0.0625) * kld(s2_out/4, s2_out_t/4, s2_out, 3) + 0.0625 * criterion(s2_out, label)
        kd4 = (1-0.0625) * kld(s1_out/4, s1_out_t/4, s1_out, 3) + 0.0625 * criterion(s1_out, label)

        #
        kd5 = (1 - 0.0625) * kld(decode[0] / temp, decode_t[0] / temp, label, 3)
        kd6 = (1 - 0.0625) * kld(decode[1] / temp, decode_t[1] / temp, label, 3)
        kd7 = (1 - 0.0625) * kld(decode[2] / temp, decode_t[2] / temp, label, 3)
        kd8 = (1 - 0.0625) * kld(decode[3] / temp, decode_t[3] / temp, label, 3)

        kd9 = (1 - 0.0625) * kld(rlayer_features[0] / 4, rlayer_features_t[0] / 4, tlayer_features_t[0], 3)
        kd10 = (1 - 0.0625) * kld(rlayer_features[1] / 4, rlayer_features_t[1] / 4, tlayer_features_t[1], 3)
        kd11 = (1 - 0.0625) * kld(rlayer_features[2] / 4, rlayer_features_t[2] / 4, tlayer_features_t[2], 3)
        kd12 = (1 - 0.0625) * kld(rlayer_features[3] / 4, rlayer_features_t[3] / 4, tlayer_features_t[3], 3)

        kd13 = (1 - 0.0625) * kld(tlayer_features[0] / 4, tlayer_features_t[0] / 4, rlayer_features_t[0], 3)
        kd14 = (1 - 0.0625) * kld(tlayer_features[1] / 4, tlayer_features_t[1] / 4, rlayer_features_t[1], 3)
        kd15 = (1 - 0.0625) * kld(tlayer_features[2] / 4, tlayer_features_t[2] / 4, rlayer_features_t[2], 3)
        kd16 = (1 - 0.0625) * kld(tlayer_features[3] / 4, tlayer_features_t[3] / 4, rlayer_features_t[3], 3)

        # -------------new--------------------------------
        # kd17 = kld(decode[0] / temp, decode_t[0]+ adj[0] / temp, label, 3)
        # kd18 = kld(decode[1] / temp, decode_t[1]+ adj[1]  / temp, label, 3)
        # kd19 = kld(decode[2] / temp, decode_t[2]+ adj[2]  / temp, label, 3)
        # kd20 = kld(decode[3] / temp, decode_t[3] / temp, label, 3)

        cross = kd9 + kd10 + kd11 + kd12 + kd13 + kd14 + kd15 + kd16 #+ kd17 + kd18 + kd19 + kd20
        # kd = kd1 + kd2 + kd3 + kd4 + kd5 + kd6 + kd7 + kd8
        kd = kd1 + kd2 + kd3 + kd4 + kd8 + kd5 + kd6 + kd7 + cross
        # kd_loss1 = dice_loss(r1, s1_new)
        # kd_loss2 = dice_loss(d1, s1_new)
        # kd_loss3 = dice_loss(r1, r4)
        # kd_loss4 = dice_loss(d1, d4)

        # edge_loss1 = criterion1(edge, bound)  + IOU(edge, bound)
        # bloss2 = criterion1(edge2, bound)  + IOU(edge2, bound)
        # bloss3 = criterion1(edge3, bound) + IOU(edge3, bound)
        # bloss4 = criterion1(edge3, bound) + IOU(boundary, bound)
        # bloss5 = criterion1(edge4, bound) + IOU(boundary, bound)


        # loss_total = fore_loss1 + fore_loss2 + fore_loss3 + fore_loss4 +fore_loss5
        #  ----loss teacher-----------
        # loss_total = 10 * ( fore_loss6)
        # -------------------------------

        # print(fore_loss6,  kd2)
        loss_total = 10 * (  self + kd) # + kd  + fore_loss1 + fore_loss2 + fore_loss3

        time = datetime.now()

        if i % 10 == 0 :
            print('{}  epoch:{}/{}  {}/{}  total_loss:{} loss:{} '
                  '  '.format(time, epoch, epochs, i, len(train_dataloader), loss_total.item(), fore_loss6))
        loss_total.backward()
        optimizer.step()
        train_loss = loss_total.item() + train_loss


    net = net2.eval()
    eval_loss = 0
    mae = 0

    with torch.no_grad():
        for j, sampleTest in enumerate(test_dataloader):

            imageVal = Variable(sampleTest['RGB'].cuda())
            depthVal = Variable(sampleTest['depth'].cuda())
            labelVal = Variable(sampleTest['label'].float().cuda())
            # bound = Variable(sampleTest['bound'].float().cuda())

            out1 = net(imageVal, depthVal)
            # out1 = net(imageVal)
            # out1, out2, out3, out4, edge_all, edge1, edge2, edge3, edge4 = net(imageVal, depthVal)
            # out = F.sigmoid(out1[13])
            # out = F.sigmoid(out1[0])
            # out = F.sigmoid(out1[0])
            out = F.sigmoid(out1[0])
            # loss = criterion_val(out, labelVal)
            loss = criterion_val(out, labelVal)

            maeval = torch.sum(torch.abs(labelVal - out)) / (320.0*320.0)

            print('===============', j, '===============', loss.item())
    #
    #         # if j==34:
    #         #     out=out[4].cpu().numpy()
    #         #     edge = edge[4].cpu().numpy()
    #         #     out = out.squeeze()567
    #         #     edge = edge.squeeze()
    #         #     plt.imsave('/home/wjy/代码/shiyan/Net/model/ENet_mobilenet/img/out.png', out,cmap='gray')
    #         #     plt.imsave('/home/wjy/代码/shiyan/Net/model/ENet_mobilenet/img/edge1.png', edge,cmap='gray')
    #
            eval_loss = loss.item() + eval_loss
            mae = mae + maeval.item()
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = '{:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    logger.info(
        f'Epoch:{epoch+1:3d}/{epochs:3d} || trainloss:{train_loss / 1500:.8f} valloss:{eval_loss / 362:.8f} || '
        f'valmae:{mae / 362:.8f} || lr_rate:{lr_rate} || spend_time:{time_str}')

    if (mae / 362) <= min(best):
        best.append(mae / 362)
        nummae = epoch+1
        torch.save(net.state_dict(), bestpath)

    torch.save(net.state_dict(), lastpath)
    print('=======best mae epoch:{},best mae:{}'.format(nummae, min(best)))
    logger.info(f'best mae epoch:{nummae:3d}  || best mae:{min(best)}')














