#!/bin/bash
set -x
set -u

# 改写自train.sh，继续ckpts训练的脚本
now=$(date +"%Y%m%d_%H%M")

jobname="train-$1-$now"
echo "job name is $jobname"

config_file=$2
mkdir -p log
mkdir -p logs/${jobname}
cp ${config_file} logs/${jobname}

export CUDA_VISIBLE_DEVICES=0

# <核心>: 运行train.py脚本
python train.py --cfg_path ${config_file} \
  --num_gpus $3 --num_nodes $4 \
  --num_epochs $5 --batch_size 1024 --test_batch_size 512 --num_workers 4 \
  --ckpt_path $6  \
  --exp_name ${jobname} 2>&1|tee log/${jobname}.log \