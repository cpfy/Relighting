#!/bin/bash
set -x
set -u


now=$(date +"%Y%m%d_%H%M%S")
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
  --num_epochs 20 --batch_size 2048 --test_batch_size 512 --num_workers 16 \
  --exp_name ${jobname} 2>&1|tee log/${jobname}.log \