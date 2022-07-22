import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs        # keyword Variable Arguments，字典形式的关键字参数
        self.create_embedding_fn()

    # 位置编码创建fn映射
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0     # 输出维度
        if self.kwargs['include_input']:        # 原始p也加入到维度？
            embed_fns.append(lambda x : x)      # 初始化每个元素为原始输入值p
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)    # linspace = 0-max范围采样N个点; 指数采样
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)    # 线性采样
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))   # torch.sin(x * 2^i)
                out_dim += d
                    
        self.embed_fns = embed_fns  # 只封装映射函数，具体值后面使用时传入
        self.out_dim = out_dim
        
    def embed(self, inputs):
        # cat为矩阵拼接函数，-1指在最高维拼接
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    # args.i_embed参数-1表示不使用位置编码，用单位阵原样映射
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,   # 取值从2^0到2^{L-1}次方
                'num_freqs' : multires,         # 采样频次
                'log_sampling' : True,          # log采样true则2^0,2^1,...,2^{L-1}指数采样，否则在[2^0, 2^{L-1}]均匀采样
                'periodic_fns' : [torch.sin, torch.cos],    # 使用sin和cos(2^i \pi p)升维
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D                              # 网络深度depth=8
        self.W = W                              # 网络宽度width=256
        self.input_ch = input_ch                # 输入坐标X维度 = 3*2L = 60
        self.input_ch_views = input_ch_views    # 输入观测方向d维度 = 3*2L = 24（这里d用方向向量没用角度？）
        self.skips = skips                      # 第五层设置一个Concatenation，见25页原文
        self.use_viewdirs = use_viewdirs

        # nn.Linear设置网络中的全连接层输入/输出维度，ModelList组合各模块，可视为py中的list
        # 在skip的第4层shape为(W + input_ch, W)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])


        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # 全连接层：256+ch -> 128 [ReLU]
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])


        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    # NN整个前向传播过程
    def forward(self, x):
        # 整个拼接（如果有拼接）tensor在最高维分割为X、d两块
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts       # 坐标X的tensor
        for i, l in enumerate(self.pts_linears):    # enum两项分别是索引、数据
            h = self.pts_linears[i](h)              # 遍历传播8层全连接层
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)   # skip层拼接一波

        if self.use_viewdirs:   # d只用最后两层
            alpha = self.alpha_linear(h)                # 体密度sigma直接额外层256 -> 1输出
            feature = self.feature_linear(h)            # feature映射了最后一层的信息？
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs      # 4个输出rgb+alpha

    # 加载模型数据，共pts,feature,views,rgb,alpha五层Linear
    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
# 渲染时辅助投射射线用，返回公式中的 r = o + td
def get_rays(H, W, K, c2w):
    # meshgrid用于生成二维网格，此处构造 W*h 个像素
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()   # .t()用于矩阵转置
    j = j.t()

    # .stack()用同维度张量堆叠的方式生成新高维张量; .ones_like()生成与输入同大小的、全为1 tensor
    # K[0][0],K[1][1]为焦距；K[0][2],K[1][2]分别为一半长宽W,H
    # 计算得到 W*H 大小的每个像素的观测方向向量 (W/f, -H/f, -1) ，取值范围[-0.5,+0.5]
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


# 返回np.而不是torch.类型表示的o和d
def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


# NDC坐标系下的o和d坐标
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]     # 三点表示任意个双冒号切片的省略，a[...,x]等价于a[:,:,x]
    rays_o = rays_o + t[...,None] * rays_d          # [...,None]切片保留全部在最后一维升维
    
    # Projection[某种方式投影到NDC下，与f与fov的tan换算关系有关？]
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5    # prevent nans(Not a Number)
    pdf = weights / torch.sum(weights, -1, keepdim=True)    # 权重随时间分布的概率密度函数取值
    cdf = torch.cumsum(pdf, -1)     # 同一行/列/维度累加，概率密度中的F(x)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    # det=true表明perturb=0，不加入扰动
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        # rand=均匀分布；randn=标准正态分布
        # 原shape最后一维替换为N_samples
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()      # 保证tensor底层存储顺序与按行优先一维展开的元素顺序一致，而不是套着虚假的view
    inds = torch.searchsorted(cdf, u, right=True)   # 在排序后的cdf中搜索u中元素的索引值，right表示同值时往右贪心取大索引
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)     # unsq(1)表示在第1维上增加一维"1"
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)   # gather在输入的dim=2按inds_g索引取值

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)  # where相当于三目运算，满足从x取，否则y取
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
