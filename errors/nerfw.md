

## NeRF-W

主要参考光照渲染部分

原地址：https://github.com/kwea123/nerf_pl/tree/nerfw

在 phototourism+BG 数据集上的demo：https://github.com/kwea123/nerf_pl/releases/tag/nerfw_branden



#### Q1

`requirements.txt` 自动安装未匹配 `torch` 与 `torchtext` 版本，额外安装如下：

```python
pip install torch==1.7.1+cu110 torchvision==0.8.2 torchtext==0.8.1 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```



#### Q2

使用的数据集与NeuralRecon-W似乎可以共用，图像编号及数据索引 id 都无变化，不用下载4.0GB的新官方 `phototourism` 数据集。data访问路径：

```
../NeuralRecon-W-test/data/heritage-recon/brandenburg_gate/
```

不过后面又 `cp` 一份新的：

```
data/heritage-recon/brandenburg_gate/
```



#### Q3

`test_phototourism.ipynb` 里面的 `N_emb_xyz = 15` 参数与给的ckpt文件模型大小不匹配

改为 `N_emb_xyz = 10` 即可



#### Q4

数据集编号不匹配，估计neuconw删了一些，不知道咋解决使得对应上