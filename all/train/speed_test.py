'''
**************YiHou use the Code*********** HJK
copy from https://github.com/yuhuan-wu/MobileSal

@ARTICLE{wu2021mobilesal,
  author={Wu, Yu-Huan and Liu, Yun and Xu, Jun and Bian, Jia-Wang and Gu, Yu-Chao and Cheng, Ming-Ming},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={MobileSal: Extremely Efficient RGB-D Salient Object Detection},
  year={2021},
  doi={10.1109/TPAMI.2021.3134684}
}
'''

import torch, os
from time import time
from tqdm import tqdm

from xiugai3.student_uniformer_t import SRAA
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
try:
    from torch2trt import torch2trt, TRTModule
    trt_installed = 1
except ModuleNotFoundError:
    print("please install torch2trt for TensorRT speed test!")
    trt_installed = 0
print("loaded all packages")
# from duoduoyishan.S2MA.ImageDepthNet.ImageDepthNet import ImageDepthNet
# from duoduoyishan.CPD.model.CPD_models import CPD_VGG
# from duoduoyishan.JLDCF.JL_DCF.networks.JL_DCF import build_model
# from duoduoyishan.R3Net.model import R3Net
# from duoduoyishan.AFNET import Afnet_mobile
# from duoduoyishan.adf import config
# from duoduoyishan.adf.APMFnet import build_model

model = SRAA().cuda().eval()
# model = SwinMCNet().cuda().eval()

# We use multi-scale pretrained model as the model weights
# model.load_state_dict(torch.load('/media/sunfan/date/zy_LSNet/LSNet_kd_PRUE_0.0625/Net_epoch_best_PURE.pth'))

# x = torch.randn(20,3,224,224).cuda()
# y = torch.randn(20,3,224,224).cuda()
#
# ######################################
# #### PyTorch Test [BatchSize 20] #####
# ######################################
# for i in tqdm(range(50)):
#     # warm up
#     p = model(x,y)
#     p = p + 1
#
# total_t = 0
# for i in tqdm(range(100)):
#     start = time()
#     p = model(x,y)
#     p = p + 1 # replace torch.cuda.synchronize()
#     total_t += time() - start
#
# print("origin batch 20 FPS", 100 / total_t*20)
#
# torch.cuda.empty_cache()

x = torch.randn(1,3,256,256).cuda()
y = torch.randn(1,1,256,256).cuda()

######################################
#### PyTorch Test [BatchSize 1] #####
######################################
for i in tqdm(range(50)):
    # warm up
    p = model(x,y)
    # p = model(x)
    p = p + 1

total_t = 0
for i in tqdm(range(100)):
    start = time()
    p = model(x,y)
    # p = model(x)
    p = p + 1 # replace torch.cuda.synchronize()
    total_t += time() - start

print("origin batch 1 FPS", 100 / total_t)

torch.cuda.empty_cache()

if not trt_installed:
    exit()

######################################
#### TensorRT Test [Batch Size=1] ####
######################################
x = torch.randn(1,3,224,224).cuda()
y = torch.randn(1,3,224,224).cuda()

save_path = "/media/sunfan/date/zy_LSNet/LSNet_kd_PRUE_0.0625/Net_epoch_best_PURE_temp.pth"
if os.path.exists(save_path):
    print('loading TensorRT model', save_path)
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(save_path))
else:
    print("converting model to TensorRT format!")
    model_trt = torch2trt(model, [x,y], fp16_mode=False)
    torch.save(model_trt.state_dict(), save_path)

torch.cuda.empty_cache()
with torch.no_grad():
    for i in tqdm(range(50)):
        p = model_trt(x,y)
        p = p + 1

total_t = 0
with torch.no_grad():
    for i in tqdm(range(2000)):
        start = time()
        p = model_trt(x,y)

        p = p + 1 # replace torch.cuda.synchronize()
        total_t += time() - start
print("TensorRT batch 1 FPS", 2000 / total_t)


######################################
##### TensorRT Test [BS=1, FP16] #####
######################################
save_path = "/media/sunfan/date/zy_LSNet/LSNet_kd_PRUE_0.0625/Net_epoch_best_PURE_temp_fp16.pth"
if os.path.exists(save_path):
    print('loading TensorRT model', save_path)
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(save_path))
else:
    print("converting model to TensorRT format!")
    model_trt = torch2trt(model, [x,y], fp16_mode=True)
    torch.save(model_trt.state_dict(), save_path)
print("Completed!!")

for i in tqdm(range(50)):
    p = model_trt(x,y)
    p = p + 1

total_t = 0
for i in tqdm(range(2000)):
    start = time()
    p = model_trt(x,y)
    p = p + 1 # replace torch.cuda.synchronize()
    total_t += time() - start
print("TensorRT fp16 batch 1 FPS", 2000 / total_t)
