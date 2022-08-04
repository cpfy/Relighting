

## NW-Environment

NeuralRecon-W环境也太难配了，😭

### Installation

#### kaolin安装

安装kaolin后的报错

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.

pymc3 3.11.5 requires scipy<1.8.0,>=1.7.3, but you have scipy 1.5.2 which is incompatible.
google-colab 1.0.0 requires tornado~=5.1.0, but you have tornado 6.1 which is incompatible.
datascience 0.10.6 requires folium==0.2.1, but you have folium 0.8.3 which is incompatible.
albumentations 0.1.12 requires imgaug<0.2.7,>=0.2.5, but you have imgaug 0.2.9 which is incompatible.
Successfully installed Jinja2-3.1.2 MarkupSafe-2.1.1 Pillow-9.2.0 Werkzeug-2.2.1 flask-2.0.3 itsdangerous-2.1.2 kaolin-0.11.0 scipy-1.5.2 tornado-6.1 usd-core-22.8

WARNING: Upgrading ipython, ipykernel, tornado, prompt-toolkit or pyzmq can
cause your runtime to repeatedly crash or behave in unexpected ways and is not
recommended. If your runtime won't connect or execute code, you can reset it
with "Disconnect and delete runtime" from the "Runtime" menu.
WARNING: The following packages were previously imported in this runtime:
  [PIL,tornado]
You must restart the runtime in order to use newly installed versions.
```



colab运行环境必需tornado~=5.1.0，`pip install kaolin` 会自动把tornado更新到6.1.0

由此导致昨天莫名其妙的free invalid pointer溢出及Aborted

```
src/tcmalloc.cc:283] Attempt to free invalid pointer 0x7fffc9523058 
Aborted (core dumped)
```



正常还需要git clone 安装

```
略
```





#### pytorch-lightning安装

【git安装pytorch-lightning后的报错】

此处又降级了tensorboard、requests

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.

tensorflow 2.8.2+zzzcolab20220719082949 requires tensorboard<2.9,>=2.8, but you have tensorboard 2.9.1 which is incompatible.
google-colab 1.0.0 requires requests~=2.23.0, but you have requests 2.28.1 which is incompatible.
datascience 0.10.6 requires folium==0.2.1, but you have folium 0.8.3 which is incompatible.

Successfully installed PyYAML-6.0 aiobotocore-2.1.2 aioitertools-0.10.0 anyio-3.6.1 asgiref-3.5.2 botocore-1.23.24 commonmark-0.9.1 croniter-1.3.5 deepdiff-5.8.1 dnspython-2.2.1 email-validator-1.2.1 fastapi-0.79.0 fsspec-2022.1.0 h11-0.13.0 httptools-0.4.0 jinja2-3.0.3 jmespath-0.10.0 lightning-2022.7.18 lightning-cloud-0.5.0 ordered-set-4.1.0 orjson-3.7.8 pyDeprecate-0.3.2 pyjwt-2.4.0 python-dotenv-0.20.0 python-multipart-0.0.5 requests-2.28.1 rich-12.5.1 s3fs-2022.1.0 sniffio-1.2.0 starlette-0.19.1 starsessions-1.2.3 tensorboard-2.9.1 torchmetrics-0.9.3 urllib3-1.25.11 uvicorn-0.17.6 uvloop-0.16.0 watchgod-0.8.2 websocket-client-1.3.3 websockets-10.3
```





【pip安装pytorch-lightning==1.4.8后的报错】

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.

lightning 2022.7.18 requires tensorboard>=2.9.1, but you have tensorboard 2.8.0 which is incompatible.
Successfully installed future-0.18.2 pyDeprecate-0.3.1 pytorch-lightning-1.4.8

WARNING: The following packages were previously imported in this runtime:
  [deprecate,pytorch_lightning]
You must restart the runtime in order to use newly installed versions.
```



#### torchtext问题

**报错：**No module named 'torchtext.legacy'

罪魁祸首在于torchtext\=\=1.13.0取消了torchtext.legacy，但pytorch_lightning\=\=1.4.8仍保留以下函数段：

```
if _TORCHTEXT_AVAILABLE:
    if _compare_version("torchtext", operator.ge, "0.9.0"):
        from torchtext.legacy.data import Batch
    else:
        from torchtext.data import Batch
```



`git clone` 安装可补充大量额外包



**报错：**ImportError: cannot import name '_package_available' from 'pytorch_lightning.utilities.imports' (/usr/local/lib/python3.7/dist-packages/pytorch_lightning/utilities/imports.py)

pip install pytorch_lightning 到1.6.5版本就会这样。。。



md，到1.4.8版本就不报错了



**报错：**from torchtext.legacy.data import Batch ModuleNotFoundError: No module named 'torchtext.legacy'

（07.31更新：又遇到这个问题，且降pl版本仍报错）





pip安装+clone安装+两个包降版本（clone后打印出的版本为1.7.0rc1）

**报错：**ImportError: cannot import name 'TestTubeLogger' from 'pytorch_lightning.loggers' (/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loggers/\_\_init\_\_.py)



#### Happy End

> 仅data_generation

泪目，Generating rays and rgbs总算跑起来了

为实现此步，共需安装以下10个包

##### 一键安装

```
!pip install open3d==0.12.0
!pip install kornia==0.4.1
!pip install loguru
!pip install torch_optimizer
!pip install trimesh==3.9.1
#!pip install kaolin
!pip install cython==0.29.20
!pip install lpips==0.1.3
!pip install torchmetrics==0.7.0
!pip install yacs
!pip install test-tube==0.7.5
!pip install tornado==5.1.0		# offset kaolin's influence
```



#### test_tube问题

**报错：**找不到TestTubeLogger

```
Traceback (most recent call last):
  File "train.py", line 12, in <module>
    from pytorch_lightning.loggers import TestTubeLogger
ImportError: cannot import name 'TestTubeLogger' from 'pytorch_lightning.loggers' (/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loggers/__init__.py)
```

问题在于1.4.8有，但新的1.7.0rc1版本的 `pytorch_lightning/loggers/__init__.py` 中移除了这个：

```
if _TESTTUBE_AVAILABLE:
    __all__.append("TestTubeLogger")
```



【解决】

自己fork并粘贴1.4.8 TestTube部分修改了一遍：https://github.com/cpfy/lightning

（08.03更新：换成新的branch——v1.4.8_origin才可）



#### pl参数报错

**报错：**错误参数progress_bar_refresh_rate

```
Traceback (most recent call last):
  File "train.py", line 79, in <module>
    main(hparams, config)
  File "train.py", line 68, in main
    gradient_clip_val=0.99
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/utilities/argparse.py", line 345, in insert_env_defaults
    return fn(self, **kwargs)
TypeError: __init__() got an unexpected keyword argument 'progress_bar_refresh_rate'
```

参见pl文档说明：https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html



### Data Generation

#### brandenburg_gate

624 10.68G

670 11.28G

750 暴涨 24.45G

Killed就是因为RAM搞爆了

```
56%|█████▌    | 751/1353 [03:01<04:46,  2.10it/s]padding valid depth percentage: from 0.0017757798791290043 to 0.2 with padding 117627
sample depth percent after padding: 0.197265625

100%|██████████| 8/8 [00:00<00:00, 286.45it/s]Killed
```



大概升级也没用，52GB肯定也不够

据说换TPU能从25G -> 35G

![enter image description here](https://i.stack.imgur.com/kjNZm.png)





#### lincoln_memorial



#### palacio_de_bellas_artes

目前仅在palacio_de_bellas_artes场景发现

**报错：**索引越界

```
Traceback (most recent call last):
  File "tools/prepare_data/prepare_data_cache.py", line 176, in <module>
    dataset = dataset_dict[args.dataset_name](**kwargs)
  File "./datasets/phototourism.py", line 125, in __init__
    self.read_meta()
  File "./datasets/phototourism.py", line 696, in read_meta
    paddings_rays = rays[valid_depth, :][pad_ind]
IndexError: index is out of bounds for dimension with size 0
```



#### pantheon_exterior

占满RAM爆炸了

```
 46%|████▌     | 640/1391 [06:40<08:27,  1.48it/s]
  0%|          | 0/9 [00:00<?, ?it/s]
100%|██████████| 9/9 [00:00<00:00, 77.16it/s]tee: log/data-generation-data/heritage-recon/pantheon_exterior-20220802_062611.logKilled
: Transport endpoint is not connected
```



### RAM

经测试，BG在取 downscale=5 最终处理完全部1353点，占用4.50G





### 训练

运行脚本train.sh时

**报错：**RuntimeError: torch.cat(): expected a non-empty list of Tensors

（无关紧要，刷新就好了）





#### lightning遗留问题

应该是torch_lightning没改干净，重新修改fork分支

```
Traceback (most recent call last):
  File "train.py", line 81, in <module>
    main(hparams, config)
  File "train.py", line 73, in main
    trainer.fit(system, datamodule=data_module)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py", line 706, in fit
    self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py", line 659, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py", line 746, in _fit_impl
    results = self._run(model, ckpt_path=self.ckpt_path)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py", line 1159, in _run
    self._log_hyperparams()
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py", line 1227, in _log_hyperparams
    logger.log_hyperparams(hparams_initial)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/utilities/rank_zero.py", line 32, in wrapped_fn
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loggers/test_tube.py", line 147, in log_hyperparams
    params = self._convert_params(params)
AttributeError: 'TestTubeLogger' object has no attribute '_convert_params'
```



#### 遗留2

一处没改，新的commit已解决

```
File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/utilities/apply_func.py", line 268, in move_data_to_device
    dtype = (TransferableDataType, Batch) if _TORCHTEXT_AVAILABLE else TransferableDataType
NameError: name '_TORCHTEXT_AVAILABLE' is not defined
```



#### 文件夹重复

未知？

```
Traceback (most recent call last):
  File "tools/prepare_data/prepare_data_cache.py", line 215, in <module>
    rgbs, all_lengths, chunk_length, split_path, args, padding_index, "rgbs"
  File "tools/prepare_data/prepare_data_cache.py", line 152, in split_to_chunks
    chunks=True,
  File "/usr/local/lib/python3.7/dist-packages/h5py/_hl/group.py", line 148, in create_dataset
    dsid = dataset.make_new_dset(group, shape, dtype, data, name, **kwds)
  File "/usr/local/lib/python3.7/dist-packages/h5py/_hl/dataset.py", line 137, in make_new_dset
    dset_id = h5d.create(parent.id, name, tid, sid, dcpl=dcpl)
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "h5py/h5d.pyx", line 87, in h5py.h5d.create
ValueError: Unable to create dataset (name already exists)
```



#### torch元素类型



```
File "/content/drive/MyDrive/NeuralRecon-W-test/datasets/phototourism.py", line 762, in __getitem__
    return self.val_num
TypeError: list indices must be integers or slices, not tuple
```



#### 异常退出

```
Traceback (most recent call last):
  File "train.py", line 86, in <module>
    main(hparams, config)
  File "train.py", line 78, in main
    trainer.fit(system, datamodule=data_module)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py", line 552, in fit
    self._run(model)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py", line 917, in _run
    self._dispatch()
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py", line 985, in _dispatch
    self.accelerator.start_training(self)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/accelerators/accelerator.py", line 92, in start_training
    self.training_type_plugin.start_training(trainer)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/plugins/training_type/training_type_plugin.py", line 161, in start_training
    self._results = trainer.run_stage()
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py", line 995, in run_stage
    return self._run_train()
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py", line 1044, in _run_train
    self.fit_loop.run()
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loops/base.py", line 111, in run
    self.advance(*args, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loops/fit_loop.py", line 200, in advance
    epoch_output = self.epoch_loop.run(train_dataloader)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loops/base.py", line 111, in run
    self.advance(*args, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loops/epoch/training_epoch_loop.py", line 118, in advance
    _, (batch, is_last) = next(dataloader_iter)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/profiler/base.py", line 104, in profile_iterable
    value = next(iterator)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/supporters.py", line 668, in prefetch_iterator
    last = next(it)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/supporters.py", line 589, in __next__
    return self.request_next_batch(self.loader_iters)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/supporters.py", line 617, in request_next_batch
    return apply_to_collection(loader_iters, Iterator, next_fn)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/utilities/apply_func.py", line 96, in apply_to_collection
    return function(data, *args, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/supporters.py", line 604, in next_fn
    batch = next(iterator)
  File "/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py", line 652, in __next__
    data = self._next_data()
  File "/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py", line 1347, in _next_data
    return self._process_data(data)
  File "/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py", line 1373, in _process_data
    data.reraise()
  File "/usr/local/lib/python3.7/dist-packages/torch/_utils.py", line 461, in reraise
    raise exception
TypeError: Caught TypeError in DataLoader worker process 0.
```



#### 内存不足

```
Traceback (most recent call last):
  File "train.py", line 86, in <module>
    main(hparams, config)
  File "train.py", line 78, in main
    trainer.fit(system, datamodule=data_module)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py", line 552, in fit
    self._run(model)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py", line 917, in _run
    self._dispatch()
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py", line 985, in _dispatch
    self.accelerator.start_training(self)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/accelerators/accelerator.py", line 92, in start_training
    self.training_type_plugin.start_training(trainer)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/plugins/training_type/training_type_plugin.py", line 161, in start_training
    self._results = trainer.run_stage()
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py", line 995, in run_stage
    return self._run_train()
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py", line 1044, in _run_train
    self.fit_loop.run()
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loops/base.py", line 111, in run
    self.advance(*args, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loops/fit_loop.py", line 200, in advance
    epoch_output = self.epoch_loop.run(train_dataloader)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loops/base.py", line 111, in run
    self.advance(*args, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loops/epoch/training_epoch_loop.py", line 130, in advance
    batch_output = self.batch_loop.run(batch, self.iteration_count, self._dataloader_idx)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loops/batch/training_batch_loop.py", line 100, in run
    super().run(batch, batch_idx, dataloader_idx)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loops/base.py", line 111, in run
    self.advance(*args, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loops/batch/training_batch_loop.py", line 147, in advance
    result = self._run_optimization(batch_idx, split_batch, opt_idx, optimizer)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loops/batch/training_batch_loop.py", line 201, in _run_optimization
    self._optimizer_step(optimizer, opt_idx, batch_idx, closure)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loops/batch/training_batch_loop.py", line 403, in _optimizer_step
    using_lbfgs=is_lbfgs,
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/core/lightning.py", line 1616, in optimizer_step
    optimizer.step(closure=optimizer_closure)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/core/optimizer.py", line 206, in step
    self.__optimizer_step(closure=closure, profiler_name=profiler_name, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/core/optimizer.py", line 128, in __optimizer_step
    trainer.accelerator.optimizer_step(self._optimizer, self._optimizer_idx, lambda_closure=closure, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/accelerators/accelerator.py", line 296, in optimizer_step
    self.run_optimizer_step(optimizer, opt_idx, lambda_closure, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/accelerators/accelerator.py", line 303, in run_optimizer_step
    self.training_type_plugin.optimizer_step(optimizer, lambda_closure=lambda_closure, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/plugins/training_type/training_type_plugin.py", line 226, in optimizer_step
    optimizer.step(closure=lambda_closure, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/torch/optim/optimizer.py", line 109, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/torch/autograd/grad_mode.py", line 27, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.7/dist-packages/torch/optim/adam.py", line 118, in step
    loss = closure()
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loops/batch/training_batch_loop.py", line 235, in _training_step_and_backward_closure
    result = self.training_step_and_backward(split_batch, batch_idx, opt_idx, optimizer, hiddens)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loops/batch/training_batch_loop.py", line 536, in training_step_and_backward
    result = self._training_step(split_batch, batch_idx, opt_idx, hiddens)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/loops/batch/training_batch_loop.py", line 306, in _training_step
    training_step_output = self.trainer.accelerator.training_step(step_kwargs)
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/accelerators/accelerator.py", line 193, in training_step
    return self.training_type_plugin.training_step(*step_kwargs.values())
  File "/usr/local/lib/python3.7/dist-packages/pytorch_lightning/plugins/training_type/training_type_plugin.py", line 172, in training_step
    return self.model.training_step(*args, **kwargs)
  File "/content/drive/MyDrive/NeuralRecon-W-test/lightning_modules/neuconw_system.py", line 381, in training_step
    self.train_level, self.sdf_threshold, device=rays.device
  File "/content/drive/MyDrive/NeuralRecon-W-test/lightning_modules/neuconw_system.py", line 295, in octree_update
    expand=False,
  File "/content/drive/MyDrive/NeuralRecon-W-test/tools/prepare_data/generate_voxel.py", line 157, in gen_octree
    octree = spc.unbatched_points_to_octree(quantized_pc, level)
  File "/content/drive/MyDrive/NeuralRecon-W-test/kaolin/kaolin/ops/spc/points.py", line 75, in unbatched_points_to_octree
    return _C.ops.spc.points_to_octree(points.contiguous(), level)
RuntimeError: CUDA out of memory. Tried to allocate 1.00 GiB (GPU 0; 14.76 GiB total capacity; 11.90 GiB already allocated; 369.75 MiB free; 13.31 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

