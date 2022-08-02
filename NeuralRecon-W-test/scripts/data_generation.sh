#!/bin/bash
set -x
set -u
set -e

now=$(date +"%Y%m%d_%H%M%S")
echo "working directory is $(pwd)"
jobname="data-generation-$1-$now"

# 该环境变量来限制CUDA程序所能使用的GPU设备，从0号编号开始！根本没有5个gpu
# export CUDA_VISIBLE_DEVICES=5

export CUDA_VISIBLE_DEVICES=0

# 为4个场景生成cache，目录为cache_sgs/
# scenes=brandenburg_gate lincoln_memorial palacio_de_bellas_artes pantheon_exterior trevi_fountain
dataset_name="phototourism"
cache_dir="cache_sgs"
root_dir=$1
min_observation=-1

if [ ! -f $root_dir/*.tsv ]; then
    python tools/prepare_data/prepare_data_split.py \
    --root_dir $root_dir \
    --num_test 10 \
    --min_observation $min_observation --roi_threshold 0 --static_threshold 0
fi
python tools/prepare_data/prepare_data_cache.py \
--root_dir $root_dir \
--dataset_name $dataset_name --cache_dir $cache_dir \
--img_downscale 1 \
--semantic_map_path semantic_maps --split_to_chunks 64 \
2>&1|tee log/${jobname}.log

# tee指令用于读取标准输入，将内容输出成文件
