

## NeRF-W

### Abstract

对传统NeRF扩展到NeRF-W，从无约束的室外图片进行视图合成地标建筑

### 1.Intro

NeRF无法处理运动物体、多样光照

第一步，建模每张图片的光照、天气等信息，在低维空间处理？使用Generative Latent Optimization框架，学习共享外观特征；

第二步，图像视为shared与image-dependent的结合，拆解为static与transient两部分。对transient我们采用二次体积辐射场（secondary volumetric radiance field）与数据独立不确定场（data-dependent uncertainty field）结合的方式，后者捕获噪声并降低动态物体对静态的干扰，最后仅用静态部分合成

我们的方法有光滑的外表插值与时间连续性，甚至在相机尺度大幅变化下，NeRF-W对于多样外观及瞬态遮挡处理很好

### 2.相关工作

* 新视图合成：**SfM**（Structure-from-Motion）和 bundle adjustment 两方法重建稀疏点云、复原相机参数，其余方法需要密集场景捕获
* 神经渲染：可学习潜藏纹理、点云、体素。与Neural Rendering Wild类似，但允许大幅相机移动

### 3.背景

#### 基本设定

NeRF的离散版渲染公式（略）

NeRF使用ReLU MLP表示体密度、颜色：
$$
[\sigma(t),z(t)]=\text{MLP}_{\theta_1}(\gamma_x(r(t))),\\
c(t)=\text{MLP}_{\theta_2}(z(t),\gamma_d(d))
$$

* 设 $\theta=[\theta_1,\theta_2]$ ，固定的函数 $\gamma_x$ 编码坐标，$\gamma_d$ 编码观察方向
* 由于体密度必须非负，且color在 $[0,1]$ 区间，最终生成 $\sigma(t),c(t)$ 时分别用ReLU、Sigmoid激活函数
* 上述写法中 $z(t)$ 将过程拆分为2个MLP，表明 $\sigma(t)$ 与观测方向 $d$ 是独立的

#### 图片表示

* 一张RGB图片表示为
  $$
  \{\mathcal{I}_i\}_{i=1}^N,\ \mathcal{I}_i\in[0,1]^{H\times W\times 3}
  $$

* 每个图片 $\mathcal{I}_i$ 对应一组相机内外参数，该参数可用 **SfM** 估计

* 预计算图片 $i$ 中的像素 $j$ 对应的相机射线集合 $\{r_{ij}\}_{j=1}^{H\times W\times 3}$ ，射线的方向 $d_{ij}$ 且 $r_{ij}(t)=o_i+td_{ij}$

#### 优化与Loss

同时优化两个MLP，**coarse** 及 **fine** 版本，粗糙模型预测的density用于fine模型正交采样？

通过如下损失函数同时优化两个模型：
$$
\sum_{ij}\left\| C(r_{ij})-\hat{C}^c(r_{ij}) \right\|_2^2+
\left\| C(r_{ij})-\hat{C}^f(r_{ij}) \right\|_2^2
$$
$C(r_{ij})$ 指图像 $\mathcal{I}_i$ 中的光线 $j$ 的颜色，$\hat{C}^c$ 、 $\hat{C}^f$ 分别指粗糙与精细模型

#### 补充

SfM的过程：

1. 获得多视角图片
2. 特征点的检测与匹配
3. 估计相机运动，计算相机视角相对变换
4. 已知变换，求解特征点深度
5. BA（Bundle Adjustment）求解PnP问题优化相机位姿和特征点位置

### 4.NeRF-W

设计两点优化处理无约束图片的两个问题：1、**Photometric variation**；2、**Transient objects**



#### 潜在外观建模

允许图像依赖的外观和照明变化，令图像之间的光度差异可以被显式建模

* 使用Generative Latent Optimization (**GLO**) 方法适应不同光照
* 每个图片 $\mathcal{I}_i$ 分配一个长度为 $n^{(a)}$ 的真实值外观嵌入向量 $\mathcal{l}_i^{(a)}$ ，改写辐射场方程，c变为**image-independent**的ci。$\{\mathcal{l}_i^{(a)}\}_{i=1}^N$ 随着训练优化

$$
\hat{C}_i(r)=\mathcal{R}(r,c_i,\sigma)\\
c_i(t)=\text{MLP}_{\theta_2}(z(t),\gamma_d(d),\mathcal{l}_i^{(a)})
$$

* 用这一嵌入向量，既能变化场景幅射光，又能保证3D几何信息静态不变
* 设定 $n^{(a)}$ 较小，使得连续光照插值？

#### 瞬态物体

允许对瞬态物体进行联合估计，并从三维世界的静态表示中分离出来

* 分离sigma、color函数，分别充当static head、transient head，易于分离occluder。丢弃不可靠的像素
* 改写体渲染函数，体密度与辐射c拆分出瞬态部分：

$$
\hat{C}_i(r)=\sum_{k=1}^KT_i(t_k)(\alpha(\sigma(t_k)\delta_k)c_i(t_k)+\alpha(\sigma_i^{(\tau)}(t_k)\delta_k)c_i^{(\tau)}(t_k))\\
T_i(t_k)=\exp{\left(-\sum_{k'=1}^{k-1}\left(\sigma(t_{k'})+\sigma_i^{(\tau)}(t_{k'})\right)\delta_{k'}\right)}
$$

* 瞬态成分变化及排除对静态干扰（待补充MLP公式3）

##### 第三项MLP

使用贝叶斯学习框架对颜色的不确定部分建模，假设观察到的像素强度是嘈杂无序的，噪声仅与输入相关（异方差的），我们认为观察颜色 $C_i(r)$ 各向同性的正态分布，该分布具有取决于图片和光线的方差 $\beta_i(r)^2$ 及平均值 $\hat{C}_i(r)$ 

这一方差有与颜色类似的alpha-合成方式，基于瞬态密度 $\sigma_i^{(\tau)}(t)$ ：
$$
\hat{\beta}_i(r)=\mathcal{R}(r,\beta_i,\sigma_i^{(\tau)})
$$
 为使得瞬态模块对于不同图片变化，为每张训练图  $\mathcal{I}_i$ 分配第二个嵌入向量 $\mathcal{l}_i^{(\tau)}\in\R^{n^{(\tau)}}$  ，作为瞬态MLP的输入：
$$
[\sigma_i^{(\tau)}(t),c_i^{(\tau)}(t),\tilde{\beta}_i(t)]=\text{MLP}_{\theta_3}\left(z(t),\mathcal{l}_i^{(\tau)}\right)\\
\beta_i(t)=\beta_\min+\log\left(1+\exp\left(\tilde{\beta_i(t)}\right)\right)
$$
分别对sigma和c用ReLU、sigmoid激活函数，对 $\beta_i(t)$ 用softplus作为激活函数

参见模型整体架构如下：

<img src="https://longtimenohack.com/posts/paper_reading/2021_martin_nerfw/image-20210721210508801.png" alt="NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo  Collections - Jianfei Guo" style="zoom:50%;" />

##### L损失函数

对于光线r，图片i，已知真实颜色 $C_i(r)$ ，损失函数为：
$$
L_i(r)=\frac{\left\| C_i(r)-\hat{C}_i(r) \right\|_2^2}{2\beta_i(r)^2}+
\frac{\log\beta_i(r)^2}{2}+
\frac{\lambda_u}{K}\sum_{k=1}^K\sigma_i^{(\tau)}(t_k)
$$
参数解释：

* 前两项为 $C_i(r)$ 的负对数似然，由正态分布及均值 $\hat{C}_i(r)$ 、方差 $\beta_i(r)^2$ 导出

* 较大的方差 $\beta_i(r)^2$ 会减弱分配给像素的重要性？第一项与第二项互相制衡
* 第三项是在瞬态密度 $\sigma_i^{(\tau)}(t)$ 上的L1正则化矩阵，并与lambda相乘。避免模型用瞬态密度解释静态现象



最终测试时扔掉transient部分，仅渲染 $\sigma(t)$ 及 $c(t)$ 

![不同MLP部分示意](https://user-images.githubusercontent.com/30110832/179965091-5ba90ed3-e12b-40ad-95e3-d1dd55a6e130.png)

#### 优化

使用 **coarse-fine** 双模型，fine模型用上文模型及loss，coarse模型只使用潜在外观建模模块。训练时同时优化外观、瞬态嵌入向量

损失函数略变化
$$
\sum_{ij}L_i(r_{ij})+
\frac{1}{2}\left\| C(r_{ij})-\hat{C}^c(r_{ij}) \right\|_2^2
$$
四个超参数：$\lambda_u,\beta_{\min}$ ，嵌入向量维度 $n^{(a)},n^{(\tau)}$

### 5.实验

与NRW、NeRF、NeRF-A、NeRF-W两个消融实验对比



### 6.结论与不足

* 训练集中缺少的部分渲染质量下降；对相机校准敏感，会导致模糊
* 处理不同光照及transient，室外重建效果很好



### 参考

[0] 主页+代码

> https://nerf-w.github.io/
>
> https://github.com/kwea123/nerf_pl

[1] 补充解读（奇怪的自媒体）

> https://zhuanlan.zhihu.com/p/172050258

[2] SfM补充

> 原文：Johannes Lutz Schönberger and Jan-Michael Frahm. Structure-from-motion revisited. CVPR, 2016.
>
> 解读1（仅推导）：https://zhuanlan.zhihu.com/p/78533248
>
> 解读2（全流程）：https://zhuanlan.zhihu.com/p/492570938