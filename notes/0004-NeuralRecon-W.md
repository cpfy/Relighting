

## NeuralRecon-Wild

### Abstract

NeuralRecon-Wild，从多种不同光照下网络旅游图片中重建表面。提出混合**voxel-guided**、**surface-guided**两种采样方法，带来更高的重建质量；提出评估室外重建性能的基准与协议

### 1.Intro

* 真实世界中的旅游景点图片丰富，但粒度等多变、光照不一致，需要相当的鲁棒性

* 受NeRF-W启发，引入appearance embeddings，但以meshes作为输出，而不是辐射场。Meshes提供场景几何的直接表示，且可以容易地引入标准图形管线中
* 借鉴NeuS体渲染方法，但原方法计算量太大，每个场景用32个GPU训练了10天？？？
* 改进传统采样策略，体积引导 + 表面引导的混合采样

<img src="https://pic4.zhimg.com/v2-29f35199bdc750358ffab58853b706c7_r.jpg" alt="preview"  />

### 2.相关工作

* 多视图重建：MVS、点云、深度、Mesh等等各种方法

* 室外重建：过去在SfM、MVS任务用到；NeRF-W无法高质量表面重建
* 神经隐式表达：广泛用于造型生成、视图合成、相机姿态估计、内分解（？）。分为两类：
  * Surface rendering：精确处理几何表面，但需要地面真值遮罩等限制
  * Volume rendering：复杂场景精确，但无法精确处理表面几何
  * 一些工作尝试融合，重点是novel view，我们基于NeuS扩展使其能接受无约束相机角度，专注模型几何

### 3.方法

结合借鉴NeRF-W对无约束图片处理，扩展NeuS几何表面表示

【沿用定义】三维点x，观察方向v，取d为SDF函数（表面为S={x|d(x)=0}）；c为颜色；$\{e_i\}_{i=1}^N$ 是apperance embedding，随训练过程一起更新
$$
d=\text{MLP}_{\text{SDF}}(x)\\
c_i=\text{MLP}_{\text{COLOR}}(x,v,e_i)
$$
射线r(t)=o+tv，图像i的颜色
$$
\hat{C}_i(r)=\int_0^{+\infty}w(t)c_i(r(t),v,e_i)dt
$$
为重建几何，需要与NeRF-W的transient head区分动态或静态场景不同的方法，见3.2节

#### 高效采样

##### 体素引导采样

* 只保留表面附近采样点

* 用SfM估计相机姿态，输出稀疏点云（**sparse volume**） $V_{\text{sfm}}$ ，通过3D dilation操作保证volume包含大多可见表面

* 此阶段在between ray and $V_{\text{sfm}}$ 的交点附近采样 $n_v$ 个点
* constructed sparse voxel能粗糙地分为前景、背景区域，移除与稀疏体素不相交光线以减少30%计算量

##### 表面引导采样

* NeuS多轮迭代至fine-level方案过于耗时

* 提出一种增加真实表面附近采样密度方案：缓存每一轮次**sparse voxels**内的SDF预测值 $V_{\text{cache}}$ ，每轮利用它查询表面位置
* $V_{\text{cache}}$ 是基于 $V_{\text{sfm}}$ 建立的深度为 $l$ 的八叉树，定期更新保证SDF值最新
* 在查询到的表面位置 $\hat{x}$ 附近区间 $(\hat{x}-t_s,\hat{x}+t_s)$ 采样 $n_s$ 个点

##### 采样混合

由于voxel采样范围大、更稀疏，最终每条光线采样 $n_v+2n_s$ 个点

#### 其它细节

* 如果用NeRF-W的transient NeRF head，由于收敛快相比几何 MLP d 会占据支配地位

* 监督信号，使用了类似NeuS的 $\mathcal{L}_{\text{COLOR}}$ 和 $\mathcal{L}_{\text{REG}}$ 
* 无纹理的天空无运动视差，在 $V_{\text{sfm}}$ 中的背景射线（绝大部分是天空）被标记在语义 mask 中，用  $\mathcal{L}_{\text{MASK}}$ 处理并赋予较小权重 

### 4.Heritage-Recon基准

提出的真实3D几何数据集Heritage-Recon，取材于Open Heritage 3D

### 5.实验

#### 参数细节

体素引导采样5000次迭代，之后加入表面引导采样

MLP结构：几何MLP，8层、512隐藏单元；颜色MLP，4层，256隐藏单元

$V_{\text{sfm}}$ 体素大小对BG、LM、PE、PBA场景分别是xx

八叉树 $V_{\text{cache}}$ 深度统一取 $l=10$

采样上下区间半径定义为体素大小s的 $t_s=16/2^l$ 倍

采样点数使用 $n_v=8$ 及 $n_s=8$ 

#### 对比

与NeRF-W、Vis-MVS、COLMAP对比，基本第一第二

* 体积渲染方法：NeRF-W ；

- 基于学习 MVS 方法：Vis-MVSNet ；

- 传统 MVS 方法：COLMAP (patch-match) ；

#### 评估

评估每个场景的precision, recall, and F1 scores。**area under the curve (AUC)**

时间比Colmap、NeRF-W等快2-7倍

消融实验表明Hybrid比单独Voxel或Sphere都更有效

### 6.结论与不足

不可见区域的几何不准确

hybrid提升准确度；一些优化加速方案

展望：逆向绘制不同时间尺度下的动态场景绘制



### 参考

[0] 主页+代码

> https://zju3dv.github.io/neuralrecon-w/
>
> https://github.com/zju3dv/NeuralRecon-W

[1] 补充解读

> https://zhuanlan.zhihu.com/p/531009053

[2] 有趣的点云数据Heritage

> https://openheritage3d.org/project.php?id=d51v-fq77



#### 代码/实现相关

[1] yacs库配置实验超参数

> https://zhuanlan.zhihu.com/p/366289700

[2] 优秀的Pytorch-Lighting库及使用

> https://zhuanlan.zhihu.com/p/353985363
>
> 官方文档：https://pytorch-lightning.readthedocs.io/
>
> cang'khttps://github.com/Lightning-AI/lightning

[3] 图像质量评价指标 PSNR 和 SSIM

> https://zhuanlan.zhihu.com/p/50757421

[4] kaolin库

有效几何函数操控3D内容：主要用于将以多边形网格、点云、有符号距离函数或体元栅格形式实现的张量3D数据集装入PyTorch

> https://developer.nvidia.com/zh-cn/blog/new-nvidia-kaolin-library-release-streamlines-3d-deep-learning-research-workflows/
>
> https://zhuanlan.zhihu.com/p/91988486

[5] 八叉树octree

> https://www.zhihu.com/question/25111128

[6] datasets/data.py 下 os.walk() 函数

> https://zhuanlan.zhihu.com/p/149824829

[7] num_workers、batch_size等参数设置

> https://blog.csdn.net/qq_24407657/article/details/103992170

[8] Eikonal Loss 程函方程（Renderer 773 行左右）

不过貌似不是同一个东西

> https://en.wikipedia.org/wiki/Eikonal_equation
>
> https://zh.m.wikipedia.org/zh-hans/%E7%A8%8B%E5%87%BD%E6%96%B9%E7%A8%8B

[9] 关于python的Dunder Methods

> https://www.tutorialsteacher.com/python/magic-methods-in-python
>
> https://zhuanlan.zhihu.com/p/344951719
