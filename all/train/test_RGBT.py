import torch as t
# from RGBT_dataprocessing_CNet import testData1,testData2,testData3
from RGBT_dataprocessing_CNet import testData1
from torch.utils.data import DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
import  numpy as np
from datetime import datetime
import cv2
from thop import profSile
test_dataloader1 = DataLoader(testData1, batch_size=1, shuffle=False, num_workers=4)
# test_dataloader2 = DataLoader(testData2, batch_size=1, shuffle=False, num_workers=4)
# test_dataloader3 = DataLoader(testData3, batch_size=1, shuffle=False, num_workers=4)
from tqdm import tqdm
from time import time
from xiugai3.ablation_student.student_uniformer_t_backbone import SRAA

net = SRAA()
net.load_state_dict(t.load('/media/user/shuju/Pth/wanzhen_rail_2023_04_22_16_23_last.pth'))   ########gaiyixia

a = '/home/user/Documents/wby/SalMap/'
b = 'Fourth_wanzhen'  ##########gaiyixia
c = '/rail_362/'
d = '/VT1000/'
e = '/VT5000/'

aa = []

vt800 = a + b + c
vt1000 = a + b + d
vt5000 = a + b + e


path1 = vt800
isExist = os.path.exists(vt800)
if not isExist:
	os.makedirs(vt800)
else:
	print('path1 exist')

with torch.no_grad():
	net.eval()
	net.cuda()
	test_mae = 0

	for i, sample in enumerate(test_dataloader1):
		image = sample['RGB']
		depth = sample['depth']
		label = sample['label']
		name = sample['name']
		name = "".join(name)

		image = Variable(image).cuda()
		depth = Variable(depth).cuda()
		label = Variable(label).cuda()

		out1 = net(image, depth)
		out = torch.sigmoid(out1[0])
		# out = label
		out_img = out.cpu().detach().numpy()
		out_img = out_img.squeeze()
		plt.imsave(path1 + name + '.png', arr=out_img, cmap='gray')

		# out = label
		# # print(out.shape)
		# out = F.interpolate(out, size=(320, 320), mode='bilinear', align_corners=False)
		# out_img = out.cpu().detach().numpy()
		# out_img = np.max(out_img, axis=1).reshape(320, 320)
		# out_img = (((out_img - np.min(out_img))/(np.max(out_img) - np.min(out_img)))*255).astype(np.uint8)
		# out_img = cv2.applyColorMap(out_img, cv2.COLORMAP_JET)
		# cv2.imwrite(path1 + name + '.png', out_img)
		print(path1 + name + '.png')

	a = torch.randn(1, 3, 320, 320).cuda()
	b = torch.randn(1, 1, 320, 320).cuda()
	flops, parameters = profile(net, (a, b))
	print(flops / 1e9, parameters / 1e6)

##########################################################################################


# path2 = vt1000
# isExist = os.path.exists(vt1000)
# if not isExist:
# 	os.makedirs(vt1000)
# else:
# 	print('path2 exist')
#
# with torch.no_grad():
# 	net.eval()
# 	net.cuda()
# 	test_mae = 0
# 	prec_time = datetime.now()
# 	for i, sample in enumerate(test_dataloader2):
# 		image = sample['RGB']
# 		depth = sample['depth']
# 		label = sample['label']
# 		name = sample['name']
# 		name = "".join(name)
#
# 		image = Variable(image).cuda()
# 		depth = Variable(depth).cuda()
# 		label = Variable(label).cuda()
#
#
# 		# out1,out2,out3,out4,out5 = net(image, depth)
# 		# out1, out2 = net(image, depth)
# 		out1, out2, out3, out4, out5, out6, out7, out8 = net(image, depth)
# 		out = torch.sigmoid(out1)
#
#
# 		out_img = out.cpu().detach().numpy()
# 		out_img = out_img.squeeze()
#
# 		plt.imsave(path2 + name + '.png', arr=out_img, cmap='gray')
# 		print(path2 + name + '.png')
# 	cur_time = datetime.now()




#######################################################################################################
#
# path3 = vt5000
# isExist = os.path.exists(vt5000)
# if not isExist:
# 	os.makedirs(vt5000)
# else:
# 	print('path3 exist')
#
# with torch.no_grad():
# 	net.eval()
# 	net.cuda()
# 	test_mae = 0
# 	prec_time = datetime.now()
# 	for i, sample in enumerate(test_dataloader3):
# 		image = sample['RGB']
# 		depth = sample['depth']
# 		label = sample['label']
# 		name = sample['name']
# 		name = "".join(name)
#
# 		image = Variable(image).cuda()
# 		depth = Variable(depth).cuda()
# 		label = Variable(label).cuda()
#
#
# 		# out1,out2,out3,out4,out5= net(image, depth)
# 		# out1, out2 = net(image, depth)
# 		out1, out2, out3, out4, out5, out6, out7, out8 = net(image, depth)
# 		out = torch.sigmoid(out1)
#
#
#
#
#
# 		out_img = out.cpu().detach().numpy()
# 		out_img = out_img.squeeze()
#
#
# 		plt.imsave(path3 + name + '.png', arr=out_img, cmap='gray')
# 		print(path3 + name + '.png')
#
# 	cur_time = datetime.now()
#   TIANCAIDAOCIYIYOU








