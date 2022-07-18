## NeuralRecon

### Abstract

NeuralRecon，从单目相机视频三维场景重建，截断符号距离函数（**TSDF，Truncated Signed Distance**
**Function**）。实时性、准确性均++

### 1.Intro

* 传统基于深度TSDF：未利用帧连续性，多次计算不一致且浪费
* 增量重建；coarse-fine设计；GRU（Gated Recurrent Unit）模块复用信息？

### 3.方法

#### 系统架构

<img src="https://zju3dv.github.io/neuralrecon/images/neucon-arch.png" alt="NeuralRecon Architechture" style="zoom: 33%;" />

#### 关键帧选取

* 位移超过 $t_{\text{max}}$ 、角度超过 $R_{\text{max}}$ 选为新关键帧
* 每个视图中以固定的最大深度范围dmax计算出一个包含所有关键帧视图的FBV（fragment bounding volume）。在重建每个片段的过程中，只考虑FBV内的区域

#### 拼接与融合

* Image Feature Volume Construction：每个体素可见性权重平均计算
* Coarse-to-fine TSDF Reconstruction：由粗到细TSDF体
* GRU Fusion：利用之前片段重建的基础
  * （此处四个递推公式）

* Integration to the Global TSDF Volume
* Supervision

### 4.实验

数据集、基准等介绍

比较对象：MVDepthNet、GPMVS、DPSNet、COLMAP等等

消融实验：证明GRU Fusion有效性；view片段取N=9效果最好

### 5.结论

略

### 参考

[0] 主页+代码

> https://zju3dv.github.io/neuralrecon/
>
> https://github.com/zju3dv/NeuralRecon