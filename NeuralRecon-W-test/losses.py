import torch
from torch import nn
class NeuconWLoss(nn.Module):
    """
    Equation 13 in the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
        s_l: sigma loss (3rd term in equation 13)
    """
    def __init__(self, coef=1, igr_weight=0.1, mask_weight=0.1, depth_weight=0.1, floor_weight=0.01, config=None):
        super().__init__()
        self.coef = coef
        self.igr_weight = igr_weight
        self.mask_weight = mask_weight
        self.depth_weight = depth_weight
        self.floor_weight = depth_weight
        
        self.config = config

    def forward(self, inputs, targets, masks=None):
        ret = {}
        if masks is None:
            masks = torch.ones((targets.shape[0], 1)).to(targets.device)
        mask_sum = masks.sum() + 1e-5

        # 第一项color_loss计算
        # l1_loss的文档：https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
        # l_n=|x_n-y_n| 后面的sum表示 L=\sum l_n，最终计算结果是指每个像素的颜色误差之和
        # masks是torch.ones()生成的，因此此处即表示 m*n 或者说 i*j
        color_error = (inputs['color'] - targets) * masks
        ret['color_loss'] = torch.nn.functional.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum

        # 第二项normal_loss计算
        # 系数0.1 * （估计是网络算出的相关参数），后面同理
        ret['normal_loss'] = self.igr_weight * inputs['gradient_error'].mean()

        if self.config.NEUCONW.MESH_MASK_LIST is not None:
            ret['mask_error'] = self.mask_weight * inputs['mask_error'].mean()

        if self.config.NEUCONW.DEPTH_LOSS:
            ret['sfm_depth_loss'] = self.depth_weight * inputs['sfm_depth_loss'].mean()

        if self.config.NEUCONW.FLOOR_NORMAL:
            ret['floor_normal_error'] = self.floor_weight * inputs['floor_normal_error'].mean()

        for k, v in ret.items():
            ret[k] = self.coef * v

        return ret

loss_dict = {'neuconw': NeuconWLoss}