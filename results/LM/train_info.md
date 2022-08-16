

## Lincoln Memorial

08.13上午调好、启动训练。

亲测ds=2只能读取到938/1093；ds=3最高峰时占用14.42G尚可接受

batch_size调回OSR也推荐的12、24G内存GPU用的2048



### Epoch

1. train-LM-20220813_0233：iter 0-66000
1. train-LM-20220813_1512：iter 68000-82000
1. train-LM-20220814_0222：8.4w-19.4w
1. train-LM-20220815_0218：19.6w-23.2w
1. train-LM-20220815_1450：23.4w-30w





### Cache

占用大小比之前 `img_downscale=5` 大了很多，前1-5样本大小如下：

```
 12%|█▏        | 126/1093 [00:20<02:03,  7.82it/s]【Size：rays】torch.Size([143325, 12])
【Len：all_rays】1
【Size：rays】torch.Size([204102, 12])
【Len：all_rays】2
【Size：rays】torch.Size([210535, 12])
【Len：all_rays】3
【Size：rays】torch.Size([211068, 12])
【Len：all_rays】4
【Size：rays】torch.Size([187083, 12])
【Len：all_rays】5
【Size：rays】torch.Size([210002, 12])
```



#### Train

每轮Epoch的总训练iter也大了非常多，达到9w，预计时长12h+

训练运行时RAM占用达到：21.30GB/25.46GB

```
Epoch 0:   1%|          | 832/90461 [06:50<12:16:43,  2.03it/s, loss=0.474, train/color_loss=0.317, train/normal_loss=0.00289, train/mask_error=0.154, train/psnr=16.60]
```



最后的epoch5（iter30w）

```
sdf filtered points 5165446, max sdf: -0.3522343039512634, min sdf: 0.27973219752311707
Update successful!!
Epoch 5:  76%|███████▌  | 68630/90461 [19:41:15<6:15:45,  1.03s/it, loss=0.237, train/color_loss=0.0803, train/normal_loss=0.00123, train/mask_error=0.143, train/psnr=25.10, val/psnr=15.70]
```

