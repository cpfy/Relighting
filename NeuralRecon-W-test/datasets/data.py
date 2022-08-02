import pytorch_lightning as pl
from datasets import dataset_dict
from torch.utils.data import DataLoader
from torch import distributed as dist
from loguru import logger
import numpy as np
import os


# [Colab]LightningDataModule一直报错。（更新：pl装1.4.8版本就成了！）
class DataModule(pl.LightningDataModule):
    """
    For distributed training, each training process is assigned
    only a part of the training rays to reduce memory overhead.
    """

    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.dataset_config = config.DATASET    # config/defaults.py中参数
        self.specific_dataset_config = getattr(
            self.dataset_config, self.dataset_config.DATASET_NAME.upper()
        )

        # 3.loader parameters
        self.train_loader_params = {
            "batch_size": args.batch_size,
            "shuffle": True,
            "num_workers": args.num_workers,
            "pin_memory": getattr(args, "pin_memory", True),
        }
        self.val_loader_params = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": args.num_workers,
            "pin_memory": getattr(args, "pin_memory", True),
        }

    # 已Override父类启动模块
    def setup(self, stage):
        try:
            self.world_size = dist.get_world_size()     # 计算：worldsize=gpu*nodes
            assert self.world_size <= 64, "world size can't larger than 64"
            assert (
                64 % self.world_size == 0
            ), "world size should be a factor of 64, otherwise automatic padding will impair the performance"
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        except RuntimeError as re:
            self.world_size = 1
            self.rank = 0
            logger.warning(str(re) + " (set world_size=1 and rank=0)")

        dataset = dataset_dict[self.dataset_config.DATASET_NAME]

        # keyword参数含义
        kwargs = {
            "root_dir": self.dataset_config.ROOT_DIR,
            "img_downscale": self.specific_dataset_config.IMG_DOWNSCALE,
            "val_num": 1,
        }
        if self.dataset_config.DATASET_NAME == "phototourism":
            kwargs[
                "semantic_map_path"
            ] = self.dataset_config.PHOTOTOURISM.SEMANTIC_MAP_PATH
            kwargs["with_semantics"] = self.dataset_config.PHOTOTOURISM.WITH_SEMANTICS
        print("Initializing data loaders...")       # <08.01> 此步在ipynb成功执行

        if self.specific_dataset_config.USE_CACHE:
            if self.specific_dataset_config.IMG_DOWNSCALE == 1:
                self.train_dataset = self._setup_dataset(
                    dataset, split="train", **kwargs
                )
            else:
                self.train_dataset = dataset(
                    split="train",
                    use_cache=True,
                    cache_paths=[self.specific_dataset_config.CACHE_DIR],
                    **kwargs,
                )
        else:
            self.train_dataset = dataset(split="train", use_cache=False, **kwargs)
        self.val_dataset = dataset(split="val", use_cache=False, **kwargs)
        print("Dataloader init finished!")

    def _get_local_split(self, items: list, world_size: int, rank: int, seed: int = 6):
        """The local rank only loads a split of the dataset."""
        n_items = len(items)
        items_permute = np.random.RandomState(seed).permutation(items)
        if n_items % world_size == 0:
            padded_items = items_permute
        else:
            padding = np.random.RandomState(seed).choice(
                items, world_size - (n_items % world_size), replace=True
            )
            padded_items = np.concatenate([items_permute, padding])
            assert (
                len(padded_items) % world_size == 0
            ), f"len(padded_items): {len(padded_items)}; world_size: {world_size}; len(padding): {len(padding)}"
        n_per_rank = len(padded_items) // world_size
        local_items = padded_items[n_per_rank * rank : n_per_rank * (rank + 1)]

        return local_items

    # 初始化数据集
    def _setup_dataset(self, dataset, split, **kwargs):
        # join将目录和文件名合成一个路径
        split_path = os.path.join(self.specific_dataset_config.CACHE_DIR, "splits") # .CACHE_DIR = 'cache'

        # <defaults.py> _CN.DATASET.ROOT_DIR = None
        # os.walk返回某文件夹下所有的子目录和文件：dirpath, dirnames, filenames；后两个list、generator类型

        # Colab执行结果均为注释
        print("##### OS PATH TEST #####")
        print(os.path)  # <module 'posixpath' from '/usr/lib/python3.7/posixpath.py'>
        print(self.dataset_config.ROOT_DIR) # data/heritage-recon/brandenburg_gate
        print(split_path)   # cache_sgs/splits
        print(os.path.join(self.dataset_config.ROOT_DIR, "split"))  # data/heritage-recon/brandenburg_gate/split
        print(type(os.walk("split")))   # <class 'generator'>

        # <推测>: 此处可能因为未完成data generation，并产生相应文件夹与路径
        # 参见：https://github.com/zju3dv/NeuralRecon-W/issues/11。由于cache missing
        # splits_names = next(
        #     # 导致StopIteration报错，据说for loop结束next后无结果导致
        #     os.walk(os.path.join(self.dataset_config.ROOT_DIR, split_path))
        # )[1]

        splits_names = "standard_split_name: STD_NAME"

        local_splits_names = self._get_local_split(
            splits_names, self.world_size, self.rank
        )
        logger.info(
            f"[rank {self.rank}]: {len(local_splits_names)} npz(s) assigned. {local_splits_names}"
        )
        return dataset(
            split=split,
            use_cache=True,
            cache_paths=local_splits_names,
            split_path=split_path,
            **kwargs,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_loader_params)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.val_loader_params)
