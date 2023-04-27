
from thop import profile
from thop import clever_format
import torch
import time
from xiugai3.student_uniformer import SRAA
def CalParams(model, input_tensor, input_tensor2):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor, input_tensor2))
    flops, params = clever_format([flops, params], "%.3f")
    print('#'*20, '\n[Statistics Information]\nFLOPs: {}\nParams: {}\n'.format(flops, params), '#'*20)

def CalParams2(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, _ = profile(model, inputs=(input_tensor))
    flops, _ = clever_format([flops], "%.3f")
    print('#'*20, '\n[Statistics Information]\nFLOPs: {}\n'.format(flops), '#'*20)


def compute_speed(model, input_size, device=0, iteration=100):
    torch.cuda.set_device(device)
    cudnn.benchmark = True

    model.eval()
    # model = model.cuda()
    model = model.cpu()

    # input = torch.randn(*input_size, device=device)
    input = torch.randn(*input_size)


    for _ in range(50):
        model(input)

    print('=========Eval Forward Time=========')
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
    return speed_time, fps

