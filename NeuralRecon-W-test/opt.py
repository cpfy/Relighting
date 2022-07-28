import argparse


# 训练相关参数解析
def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, help='config path')     # 额外的cfg配置文件
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=256,
                        help='test batch size')
    parser.add_argument('--chunk', type=int, default=16*1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=16,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--num_nodes', type=int, default=1,         # num_nodes与gpu数量相乘为world-size，默认为2？
                        help='number of nodes')
    parser.add_argument('--num_workers', type=int, default=4,       # 加载data的处理者
                        help='number of workers for data_loader')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')

    parser.add_argument('--divide_lr', default=False, action="store_true",
                        help='whether to decrease lr when resume ckpt')
    parser.add_argument('--lr_divisor', type=float, default=5,
                        help='used when change lr of resuming ckpt')    

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--refresh_every', type=int, default=1,
                        help='print the progress bar every X steps')

    return parser.parse_args()