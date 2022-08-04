import os
import math
from opt import get_opts

from datasets import DataModule
from lightning_modules.neuconw_system import NeuconWSystem

# pytorch-lightning[核心为pytorch的二次封装？]
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from config.defaults import get_cfg_defaults

# para分别为arg参数、静态文件参数
def main(hparams, config):
    caches = None
    pl.seed_everything(config.TRAINER.SEED)

    # scale lr and warmup-step automatically
    # 见issues9，nodes指分布式训练中，拥有 $num_gpus 的计算机数量
    config.TRAINER.WORLD_SIZE = hparams.num_gpus * hparams.num_nodes                    # train.sh定义，目前是1*1
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * hparams.batch_size     # train.sh定义，为2048
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS     # 使用的BS与标准BS差距倍数
    config.TRAINER.SCALING = _scaling
    config.TRAINER.LR = config.TRAINER.CANONICAL_LR * _scaling      #todo 跟opt.py里的LR关系？

    # 封装整套模型与训练系统
    system = NeuconWSystem(hparams, config, caches) 

    # 封装全部phototourism数据读取
    data_module = DataModule(hparams, config)
    
    checkpoint_callback = \
        ModelCheckpoint(dirpath=os.path.join(f'ckpts/{hparams.exp_name}',
                                               '{epoch:d}'),
                        monitor='val/psnr',     # PSNR = 峰值讯噪比（Peak signal-to-noise ratio）
                        mode='max',
                        save_top_k=-1)

    logger = TestTubeLogger(save_dir="logs",
                            name=hparams.exp_name,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False)

    # 使用的数据集图片比例
    if config.DATASET.DATASET_NAME=='phototourism' and config.DATASET.PHOTOTOURISM.IMG_DOWNSCALE==1:
        replace_sampler_ddp = False
    else:
        replace_sampler_ddp = True

    # lightning优雅的封装
    # 详细参数见文档：https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=[checkpoint_callback],
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=hparams.refresh_every,     #此用法在pytorch_lightning已废弃(progress bar的刷新频率)
                      # Passing training strategies (e.g., "ddp") to accelerator has been deprecated in v1.5.0 and will be removed in v1.7.0. Please use the strategy argument instead.

                      gpus=hparams.num_gpus,
                      num_nodes=hparams.num_nodes,

                      # （08.03更新）由于lightning又换回1.4.8版本，仍使用accelerator，上面rate同
                      accelerator='ddp' if hparams.num_gpus>1 else None,    # 废弃，用strategy替代
                      # strategy='ddp' if hparams.num_gpus > 1 else None,

                      num_sanity_val_steps=1,
                      val_check_interval=config.TRAINER.VAL_FREQ,   # yaml中，而非defaults里的参数
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None,
                      replace_sampler_ddp=replace_sampler_ddp,   # need to read all data of local dataset when config.DATASET.PHOTOTOURISM.IMG_DOWNSCALE==1
                      gradient_clip_val=0.99
                      )

    trainer.fit(system, datamodule=data_module)


if __name__ == '__main__':
    hparams = get_opts()            # 获取arg参数
    config = get_cfg_defaults()     # 指定训练、模型相关参数
    config.merge_from_file(hparams.cfg_path)    # 合并额外的cfg配置文件
    # caches = setup_shared_ray_cache(hparams)
    main(hparams, config)