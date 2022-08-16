

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



### 系列2

> In NeuralRecon-W复现时的一些问题



#### Q1

相应调用 `from rendering.renderer import *` 时，关于 `kaolin` 报错：

```
/content/drive/MyDrive/NeuralRecon-W-test/tools/prepare_data/generate_voxel.py in <module>()
     11 import os
     12 from utils.colmap_utils import read_points3d_binary
---> 13 from kaolin.ops import spc
     14 import kaolin.render.spc as spc_render
     15 import yaml

ModuleNotFoundError: No module named 'kaolin.ops'
```



真离谱，原来是没有restart runtime？？？？？？？



#### Q2

在对 `dataset` 进行 `__getitem__()` 时，路径变为 `None` 找不到报错

```
【Output】image path id is 62
【Output】image name is 04800984_8342094434
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
<ipython-input-8-dda7e836ad8c> in <module>()
      1 # 默认参数5，实际随便取，就是图像编号id
----> 2 sample = dataset[51]
      3 rays = sample['rays'].cuda()
...

FileNotFoundError: [Errno 2] No such file or directory: 'data/heritage-recon/brandenburg_gate/None/04800984_8342094434.npz'
```



观察可知，在 `phototourism.py` 读取semantic_map一行时， `self.semantic_map_path` 初始化时默认为 `None` 导致的

```python
semantic_map = np.load(
	os.path.join(
		self.root_dir, f"{self.semantic_map_path}/{image_name}.npz"
	)
)["arr_0"]
```



#### Q3

`Renderer` 传入 `nerf_far_override=config.NEUCONW.NEAR_FAR_OVERRIDE` 参数时报错：

```
Traceback (most recent call last)
<ipython-input-8-7a09947b27a6> in <module>
     52     sample_range=config.NEUCONW.SAMPLE_RANGE,
     53     boundary_samples=config.NEUCONW.BOUNDARY_SAMPLES,
---> 54     nerf_far_override=config.NEUCONW.NEAR_FAR_OVERRIDE
     55 )

1 frames
/content/drive/MyDrive/NeuralRecon-W-test/rendering/renderer.py in __init__(self, nerf, neuconw, embeddings, n_samples, n_importance, n_outside, up_sample_steps, perturb, origin, radius, s_val_base, spc_options, sample_range, boundary_samples, nerf_far_override, render_bg, trim_sphere, save_sample, save_step_sample, mesh_mask_list, floor_normal, depth_loss, floor_labels)
    102 
    103         # read unit sphere origin and radius from scene config
--> 104         scene_config_path = os.path.join(spc_options["reconstruct_path"], "config.yaml")
    105         if os.path.isfile(scene_config_path):
    106             with open(scene_config_path, "r") as yamlfile:

/usr/lib/python3.7/posixpath.py in join(a, *p)
     78     will be discarded.  An empty last part will result in a path that
     79     ends with a separator."""
---> 80     a = os.fspath(a)
     81     sep = _get_sep(a)
     82     path = a

TypeError: expected str, bytes or os.PathLike object, not NoneType
```



由于 `config.py` 中的 `_CN.DATASET.ROOT_DIR` 设置为 `None` 

而 `spc_options["reconstruct_path"]` 沿用了config中的路径作为路径，修改为正确路径



#### Q4

上一问题的延申，`renderer.py` 某一行还愚蠢地检查了一下 `spc_options["reconstruct_path"]` 路径及文件是否存在，之后才给 `self.sfm_to_gt` 赋值

```
/content/drive/MyDrive/NeuralRecon-W-test/rendering/renderer.py in render(self, rays, ts, label, perturb_overwrite, background_rgb, cos_anneal_ratio)
    823             self.origin = self.origin.to(device).float()
--> 824             self.sfm_to_gt = self.sfm_to_gt.to(device).float()
    825 

AttributeError: 'numpy.ndarray' object has no attribute 'to'
```



需要把 `reconstruct_path` 写对



#### Q5

非常频繁的错误

经查，需要把数据、Model移动到device上再进行操作

```
/content/drive/MyDrive/NeuralRecon-W-test/rendering/renderer.py in render(self, rays, ts, label, perturb_overwrite, background_rgb, cos_anneal_ratio)
    851 
--> 852         a_embedded = self.embeddings["a"](ts)
    853         print(f"【Output:renderer/852】ts = {ts}")
...
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument index in method wrapper__index_selec
```



见：https://stackoverflow.com/questions/70102323/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least

> You did not move your model to `device`, only the data. You need to call `model.to(device)` before using it with data located on `device`.



此处各种报错，输入的 `rays` 和 `ts` 完全不加 `.cuda()` 也不行。最后应该是要把所有renderer中的self参数移动到GPU。另外要注意NeRF和NeuconW这些model也要全部加 `.cuda()`



#### Q6

embeddings作为dict类型直接.to(device)会报错，需要逐元素移动

写法参见：https://discuss.pytorch.org/t/module-dictionary-to-gpu-or-cuda-device/86482



#### Q7

上述步骤之后，悲催地发生了...

```
RuntimeError: CUDA out of memory. Tried to allocate 28.57 GiB (GPU 0; 15.90 GiB total capacity; 2.43 GiB already allocated; 12.85 GiB free; 2.44 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```



两种思路

1. 参考nerf_pl，在函数f处分割chunk来获取results返回值
2. 全部去除cuda在CPU进行运算



思路1最终删掉两处sdf还是不够

```
RuntimeError: CUDA out of memory. Tried to allocate 472.00 MiB (GPU 0; 15.90 GiB total capacity; 14.53 GiB already allocated; 425.75 MiB free; 14.87 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

