import torch
# optimizer
from torch.optim import SGD, Adam
import torch_optimizer as optim
# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from .visualization import *

def get_parameters(models):
    """Get all model parameters recursively."""
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else: # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters

# 根据config参数（defaults.py或者hparam），neuconw模型的optimizer初始化（直接调用torch）
def get_optimizer(config, models):
    eps = 1e-7
    parameters = get_parameters(models)
    if config.OPTIMIZER == 'sgd':
        optimizer = SGD(parameters, lr=config.LR, 
                        momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'adam':
        optimizer = Adam(parameters, lr=config.LR, eps=eps, 
                         weight_decay=config.WEIGHT_DECAY)
                         # ,capturable = True)     # 很奇怪，依然无效
    elif config.OPTIMIZER == 'radam':
        optimizer = optim.RAdam(parameters, lr=config.LR, eps=eps, 
                                weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'ranger':
        optimizer = optim.Ranger(parameters, lr=config.LR, eps=eps, 
                                 weight_decay=config.WEIGHT_DECAY)
    else:
        raise ValueError('optimizer not recognized!')

    # See：https://github.com/pytorch/pytorch/issues/80809
    # 处理error: assert not step_t.is_cuda, "If capturable=False, state_steps should not be CUDA tensors.
    # optimizer.param_groups[0]['capturable'] = True    # [无效]考虑之前optim时就设定好
    # torch==1.12.1移除这一assert，见https://github.com/pytorch/pytorch/pull/80222
    # 最终的解决方案是colab安装1.12.1，不清楚是不是lightning的bug（估计就算有bug也已经更新，而自己装的是山寨fork 1.4.8版本）

    return optimizer

def get_scheduler(hparams, config, optimizer):
    eps = 1e-8
    if config.LR_SCHEDULER == 'steplr':
        scheduler = MultiStepLR(optimizer, milestones=config.DECAY_STEP, 
                                gamma=config.DECAY_GAMMA)
    elif config.LR_SCHEDULER == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps)
    elif config.LR_SCHEDULER == 'poly':
        scheduler = LambdaLR(optimizer, 
                             lambda epoch: (1-epoch/hparams.num_epochs)**config.POLY_EXP)
    elif config.LR_SCHEDULER == 'none':
        scheduler = None
    else:
        raise ValueError('scheduler not recognized!')

    return scheduler

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_

# 与nerf_pl比较阅读可发现：变化之处就是多加了个strict参数、调整了下代码顺序、添加warning错误输出
# 经典且重要的加载ckpt函数
def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[], strict=True):
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    
    # when prefixes_to_ignore not empty, none strict mode loading
    strict = False if len(prefixes_to_ignore) > 0 else strict
    
    if not strict:
        model_dict = model.state_dict()
        model_dict.update(checkpoint_) 
        load_dict = model_dict
    else:
        load_dict = checkpoint_
        
    missing_keys, unexpected_keys = model.load_state_dict(load_dict, strict=strict)

    if len(missing_keys) > 0:
        print(f"[WARN] missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"[WARN] unexpected keys: {unexpected_keys}")   