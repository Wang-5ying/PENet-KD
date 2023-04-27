import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
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
from model.foruth_net_shunted import *
# from Yang.new3_decoder import *
import torchvision
import time
import shutil
from log import get_logger
# from  Loss.Binary_Dice_loss import BinaryDiceLoss
# from Loss.Focal_loss import sigmoid_focal_loss

import matplotlib.pyplot as plt


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
net = fourmodel_teacher()
net.load_pre('/home/user/Documents/HJK/backbone/Shunted/ckpt_B.pth')
# net.load_state_dict(torch.load('/home/hjk/文档/third_model_GCN/backbone/Shunted/ckpt_T.pth'))
net = net.cuda()

################################################################################################################
model = 'wanzhen_SD_rail_' + time.strftime("%Y_%m_%d_%H_%M")
print_network(net, model)
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

# focaloss = sigmoid_focal_loss().cuda()
# diceloss = BinaryDiceLoss().cuda()

criterion_val = BCELOSS().cuda()

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
    loss = F.mse_loss(fstudent, fteacher, reduction='mean')
    b, c, h, w = fstudent.size()
    cnt = 1.0
    tot = 1.0
    for l in [4,2,1]:
        if l >=h:
            continue
        tmpfs = F.adaptive_avg_pool2d(fstudent, (l,l))
        tmpft = F.adaptive_avg_pool2d(fteacher, (l,l))
        cnt /= 2.0
        loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        tot += cnt
    loss = loss / tot
    loss_all = loss_all + loss
    return loss_all

################################################################################################################
lr_rate = 1e-4
optimizer = optim.Adam(net.parameters(), lr=lr_rate, weight_decay=1e-3)
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
epochs = 100
################################################################################################################

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
    net = net.train()
    prec_time = datetime.now()
    for i, sample in enumerate(train_dataloader):

        image = Variable(sample['RGB'].cuda())
        # image = Variable(sample['RGB'].float().cuda())
        depth = Variable(sample['depth'].cuda())
        label = Variable(sample['label'].float().cuda())
        bound = Variable(sample['bound'].float().cuda())
        background1 = 1 - label

        optimizer.zero_grad()

        # out1, out2, out3, out4, out5 = net(image, depth)
        s4_out, s3_out, s2_out, s1_out, out_r, out_d = net(image, depth)
        # out1, out2, out3, out4, r1, d1, r4, d4, s1_new, Z4, S4, rd4 = net(image, depth)

        s4_out = F.sigmoid(s4_out)
        s3_out = F.sigmoid(s3_out)
        s2_out = F.sigmoid(s2_out)
        s1_out = F.sigmoid(s1_out)
        out_r = F.sigmoid(out_r)
        out_d = F.sigmoid(out_d)


        bs4_out = 1 - s4_out
        bs3_out = 1 - s3_out
        bs2_out = 1 - s2_out
        bs1_out = 1 - s1_out
        bout_r = 1 - out_r
        bout_d = 1 - out_d


        fore_loss1 = criterion1(s3_out, label) + IOU(s3_out, label)
        fore_loss2 = criterion1(s2_out, label) + IOU(s2_out, label)
        fore_loss3 = criterion1(s1_out, label) + IOU(s1_out, label)
        fore_loss4 = criterion1(out_r, label) + IOU(out_r, label)
        fore_loss5 = criterion1(out_d, label) + IOU(out_d, label)
        fore_loss6 = criterion1(s4_out, label) + IOU(s4_out, label)


        back_loss1 = (criterion1(bs4_out, background1) + IOU(bs4_out, background1))
        back_loss2 = (criterion1(bs3_out, background1) + IOU(bs3_out, background1))
        back_loss3 = (criterion1(bs2_out, background1) + IOU(bs2_out, background1))
        back_loss4 = (criterion1(bs1_out, background1) + IOU(bs1_out, background1))
        back_loss5 = (criterion1(bout_r, background1) + IOU(bout_r, background1))
        back_loss6 = (criterion1(bout_d, background1) + IOU(bout_d, background1))


        # sd_loss1 = hcl(r1, s4)
        # sd_loss2 = hcl(d1, s4)


        loss_total = fore_loss1 + fore_loss2 + fore_loss3 + fore_loss4 + fore_loss5 + fore_loss6  \
                     + back_loss1 + back_loss2 + back_loss3 + back_loss4 + back_loss5 + back_loss6


        time = datetime.now()

        if i % 10 == 0 :
            print('{}  epoch:{}/{}  {}/{}  total_loss:{} loss:{} '
                  '  '.format(time, epoch, epochs, i, len(train_dataloader), loss_total.item(), fore_loss6))
        loss_total.backward()
        optimizer.step()
        train_loss = loss_total.item() + train_loss


    net = net.eval()
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
            out = F.sigmoid(out1[0])
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













