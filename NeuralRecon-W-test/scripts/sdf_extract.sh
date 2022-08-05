#!/bin/bash
set -x
set -u

now=$(date +"%Y%m%d_%H%M%S")
jobname="sdf-$1-$now"
echo "job name is $jobname"

config_file=$2
ckpt_path=$3
eval_level=$4   # 示例给的10，评估等级？

python -m torch.distributed.launch --nproc_per_node=1  tools/extract_mesh.py \
--cfg_path ${config_file} \
--mesh_size 1024 --chunk 102144 \
--ckpt_path $ckpt_path \
--mesh_radius 1 --mesh_origin "0, 0, 0" --chunk_rgb 1024 --vertex_color --eval_level ${eval_level} \
2>&1|tee log/${jobname}.log

# $nproc_per_node 参数指定为当前主机创建的进程数。一般设定为当前主机的 GPU 数量。默认值：4