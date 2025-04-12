import numpy as np
import math
import torch

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
        
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def to_cuda(x: torch.Tensor | np.ndarray | tuple) -> torch.Tensor | np.ndarray | tuple:
    if isinstance(x, np.ndarray):
        if x.dtype == np.object_:
            return x
        else:
            return torch.from_numpy(x).cuda()
    elif isinstance(x, torch.Tensor):
        return x.cuda()
    elif isinstance(x, tuple):
        return tuple(to_cuda(t) for t in x)
    else:
        raise ValueError(f"Unsupported type: {type(x)}")