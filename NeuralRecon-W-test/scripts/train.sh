#!/bin/bash
set -x
set -u

# 不用这个太繁琐了
now=$(date +"%Y%m%d_%H%M")
#now=$(date +"%Y%m%d")

jobname="train-$1-$now"
echo "job name is $jobname"

config_file=$2
mkdir -p log
mkdir -p logs/${jobname}
cp ${config_file} logs/${jobname}

# 该环境变量来限制CUDA程序所能使用的GPU设备，从0号编号开始！根本没有5个gpu
# export CUDA_VISIBLE_DEVICES=5

export CUDA_VISIBLE_DEVICES=0

# <核心>: 运行train.py脚本
python train.py --cfg_path ${config_file} \
  --num_gpus $3 --num_nodes $4 \
  --num_epochs $5 --batch_size $6 --test_batch_size 512 --num_workers 4 \
  --exp_name ${jobname} 2>&1|tee log/${jobname}.log \

# $num_workers指加载数据时工作进程个数，会有各种奇葩的多线程报错。（默认值：16）
# 据说batch size大也会占用很多空间，尝试从defaults=2048降到1024
# $num_epochs 默认值20，改为2


# $num_epochs 变为可调节参数$5
# $batch_size 变为可调节参数$6，默认值2048，BG时一直用1024训练