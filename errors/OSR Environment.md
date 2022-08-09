

## NeRF OSR Environment

### Setup

（2022.08.06测试时状态）

* 不能同时申请两台高RAM GPU，不过一个高RAM+一个标准是可行的



### Installation

为运行 `scripts/train_trevi_final.sh` ，需如下安装：

```
%pip install tensorboardX==2.1
%pip install configargparse==1.2.3
```



#### SIGKILL终止

运行时的首个报错

```
Traceback (most recent call last):
  File "ddp_train_nerf.py", line 750, in <module>
    train()
  File "ddp_train_nerf.py", line 745, in train
    join=True)
  File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 240, in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
  File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 198, in start_processes
    while not context.join():
  File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 146, in join
    signal_name=name
torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with signal SIGKILL
Finished
```

