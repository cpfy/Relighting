

## NeRF-OSR

### Abstract

**NeRF-OSR**（Outdoor Scene Relighting），首个室外场景relighting方法，支持从无约束户外照片集合中同时编辑场景光照、相机视角？且支持直接对（球函数定义的）场景光照的控制

提出一种benchmark用于评估，图像质量、阴影真实性++

### 1.Intro

* 针对在VR、AR中应用的可控照明编辑（**Controllable lighting editing**）问题，需求是独立操控照明、保持几何、反射率等不变
* 现有方法针对人脸等特定物体、室内外场景；现有NeRF照明编辑的方法要么需要已知、统一的光照训练，要么对室外的投射阴影（**cast shadows**）处理的不好
* NeRF-W还不错，但缺少照明的物理解释，无法对光照、阴影参数控制
* NeRF-OSR解决这些问题，贡献：显式控制照明模块；每个场景分为占据空间+照明+阴影+漫反射率四个模块；benchmark与数据集



### 2.相关工作

#### Scene Relighting

* 与Yu and Smith[41]、Yu et al.[40]方法较相近，但其低频照明模型可导致结果不真实，我们除照明外还能编辑相机视角
* NeRF相关方法中，最相近的是NeRD，他们基于物理真实渲染估计场景的SVBRDF，缺陷是不像我们能显式建模阴影，且要求所有视角下物体的距离相近（与无约束室外照片集相悖）

#### Style-based Editing

* 此方法无需对场景照明物理理解，寻求一次编辑场景全部外观，因此缺少阴影等参数的控制
* 其方法是将某种风格（函数/映射）迁移到目标图片上



### 3.方法

输入不同时间、视角一场景RGB图片，输出某照明、视角下的结果

#### NeRF模型（回顾）

看起来一样
$$
C(\mathbf{o},\mathbf{d})=\sum_{i=1}^{N}T_i(1-\exp(-\sigma_i\delta_i))\mathbf{c}_i\\
T_i=\exp\left(-\sum_{j=1}^{i-1}\sigma_j\delta_j\right)
$$

#### 照明编码

##### 球函数（SH）NeRF

c(x) 未编码光照，重定义渲染方程：
$$
C\left(\{x_i\}_{i=1}^N,L\right)=
A\left(\{x_i\}_{i=1}^N,L\right)\odot
Lb\left(N\left(\{x_i\}_{i=1}^N,L\right)\right)
$$

* $\odot$ 符号表示逐元素相乘
* $A(x)\in\R^3$ 为累积的反射率颜色
* $L\in\R^{9\times 3}$ 为逐图片学习的球函数（SH）系数
* $b(n)\in\R^9$ 为球函数基（SH basis）
* $N(x)$ 为从累积的光线密度计算的表面法线
* 除 $b(n)$ 与 $N(x)$ 项外，其余都是可学习的

关于N的详细计算
$$
N\left(\{x_i\}_{i=1}^N,L\right)=
\frac{\hat{N}\left(\{x_i\}_{i=1}^N\right)}
{\left\|\hat{N}\left(\{x_i\}_{i=1}^N\right)\right\|^2}
$$
其中
$$
\hat{N}\left(\{x_i\}_{i=1}^N\right)=
\sum_{i=1}^N\left(\frac{\partial}{\partial x_i}\sigma(x_i)\right)\odot
T(t_i)\alpha(\sigma(x_i)\delta_i)
$$


##### 阴影生成网络

在SH基础上扩展
$$
C\left(\{x_i\}_{i=1}^N,L\right)=
S\left(\{x_i\}_{i=1}^N,L\right)
A\left(\{x_i\}_{i=1}^N,L\right)\odot
Lb\left(N\left(\{x_i\}_{i=1}^N,L\right)\right)
$$

* 阴影模型由一个经MLP $s(x,L)\in[0,1]$ 计算的标量定义，最终的阴影值由光线上累积到 $S\left(\{x_i\}_{i=1}^N,L\right)$ 计算

* 阴影预测网络接受灰度SH系数作为输入
* $L\in\R^{1\times 9}$ 而不是 $\R^{3\times 9}$ ，这是因为阴影仅与空间光照分布有关，与传统光线追踪不同



#### 目标函数

优化如下的loss函数
$$
\mathcal{L}(C,C^{(GT)},S)=MSE(C,C^{(GT)})+\lambda MSE(S,1)
$$

* $MSE(\cdot,\cdot)$ 指平均平方误差（**mean squared error**）
* 右边第一项为在颜色C上与真实值（**Ground Truth**）的重建loss
* 第二项为阴影正则化项，$\lambda$ 控制正则化强度，经实验尽量选较大、且不会降低重建图像PSNR的值
* 实验表明，移除正则化项会导致S学习到全部照明成分，除了色度/色品，从而导致SH照明无效



#### 训练细节

介绍一些训练时的方法与策略

##### 频率退火

> Frequency Annealing

初始生成的一些噪声很难操纵，难以收敛到正确的几何信息

用退火减轻这一问题，借鉴/小幅修改自Deformable NeRF的退火框架，为每个位置编码（PE,Position Encoding）附加退火系数 $\beta_k(n)$ 
$$
\gamma_k^{'}(x)=\gamma_k(x)\beta_k(n),~\text{where}\\
\beta_k(n)=\frac{1}{2}(1-\cos(\pi\text{clamp}(\alpha-i+N_{\text{fmin}},0,1)))\\
\alpha(n)=(N_{\text{fmax}}-N_{\text{fmin}})\frac{n}{N_{\text{anneal}}}
$$

* n为目前的训练轮次
* $N_{\text{fmin}}$ 为起始时PE使用频率（我用8），$N_{\text{fmax}}$ 为总的PE使用频率（我用12）

* $N_{\text{anneal}}$ 设置为 $3*10^4$
* 该策略能够大大提升几何预测表现

##### 射线方向扰动

> Ray Direction Jitter

为增强NeRF-OSR可适性，在射线的方向上应用子像素扰动，也就是说并非射到像素中心，添加一个扰动 $\psi$
$$
x_i=\mathbf{o}+t_i(\mathbf{d}+\psi)
$$
对 $\psi$ 均匀采样，使得得到的射线仍然局限在指定像素的边界内

##### 阴影网络输入扰动

对lighting的训练仍可能过拟合，为此在阴影生成网络的环境系数中添加一个微小扰动 $\epsilon$
$$
S'(\{x_i\}_{i=1}^N,L)=S(\{x_i\}_{i=1}^N,L+\epsilon)
$$
$\epsilon$ 服从分布 $\epsilon\in\mathcal{N}(0,0.0025I)$ 

该式可以理解为局部条件，即在相似的光照条件下，阴影不应该有太大的差异。这允许模型学习不同灯光之间更平滑的过渡

##### 最终实现

用NeRF++作为基础，因此仅在前景网络的单位球面范围内工作

两块RTX8000，训练 $5*10^5$ 轮次，一个含有 $2^{10}$ 射线的BS，约需要2天



#### 全部结构展示

![NeRF-OSR模型结构展示](https://user-images.githubusercontent.com/30110832/182770001-7ad21d83-7ab6-447c-867e-0b8e530c034d.png)



### 4.Benchmark

现有数据集包括PhotoTourism、MegaDepth、Yu等人[40]提出的数据集，但都有问题

我们为室外场景relighting提供新的benchmark，在规模和对真实数据进行精确数值评估的能力方面首次

我们的数据集比Yu et al.[40]大得多，包含了8个使用单反相机从不同角度拍摄的站点。在一天的不同时间，对每个站点进行多次录音。我们还捕获360◦的每个环节的环境地图。然而，与Yu等人[40]不同的是，我们明确说明了环境地图和主要记录的单反相机之间的颜色校准。为此，在测试集的每一个环节，我们还捕获“GretagMacbeth ColorChecker”色彩校准图表与单反和360◦相机同时。然后，我们应用Finlayson等人[4]的二阶方法，通过校准它们的ColorChecker值到ColorChecker来对环境地图进行颜色校正

略。。。



### 5.实验结果

#### 5.1 数据预处理

使用Tao[34]等人的分割方法移除车、人等运动体，获得高质量的masks

天空、植被区域排除



#### 5.2 Ground Truth对比

PSNR、MSE、MAE都最好；



#### 5.3 消融实验

shadows、anneals、ray jitter、shadow jitter、shadow regularariser均有效



### 6.结论与不足

* 我们的SH照明模型基于朗伯反射假设、限制低频的照明效果
* 未来设想支持：高频照明、高光、空间变化照明
* 加入discriminator提升真实性
* 多场景数据提升单场景模型
* 更好的几何、表面模型
* 阴影计算中加入明确的几何推理，可能会改善硬投影的外观



### 参考

[0] 主页+代码

>  https://4dqv.mpi-inf.mpg.de/NeRF-OSR/
>
> https://github.com/r00tman/NeRF-OSR

[1] 关于双向反射分布函数BRDF（Bidirectional Reflectance Distribution Function）

> https://zhuanlan.zhihu.com/p/21376124
>
> https://en.wikipedia.org/wiki/Bidirectional_reflectance_distribution_function
>
> https://www.zhihu.com/question/20286038