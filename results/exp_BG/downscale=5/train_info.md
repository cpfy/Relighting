

## exp_BG

与之前最主要区别为设定参数 `IMG_DOWNSCALE=5` ，以及减小了 `batch_size` 



#### Detail

1、epoch 0-1（时间5h左右）

exp_BG-20220806_084655

2、epoch 2-6

exp_BG-20220806_1454

3、epoch 7-8

train-exp_BG-20220808_0106

4、epoch 8-10

train-exp_BG-20220808_0416

5、epoch 12-15

train-exp_BG-20220808_1452

6、epoc 16-22（—iter 45w）

train-exp_BG-20220809_1452



#### Epoch 2

```
100%|██████████| 1024/1024 [00:12<00:00, 78.97it/s]max sdf: 0.21636241674423218, min sdf: -0.23751065135002136 0.21636242
start marching cubes
radius:  0.2503623632148116

Epoch 1: 100%|██████████| 23378/23378 [2:21:03<00:00,  2.76it/s, loss=0.225, train/color_loss=0.173, train/normal_loss=0.00121, train/mask_error=0.0302, train/sfm_depth_loss=0.0149, train/psnr=19.50, val/psnr=15.60]
Epoch 1: 100%|██████████| 23378/23378 [2:21:03<00:00,  2.76it/s, loss=0.225, train/color_loss=0.173, train/normal_loss=0.00121, train/mask_error=0.0302, train/sfm_depth_loss=0.0149, train/psnr=19.50, val/psnr=15.60]
INFO - 2022-08-06 13:25:00,563 - base - FIT Profiler Report

Action                             	|  Mean duration (s)	|Num calls      	|  Total time (s) 	|  Percentage %   	|
--------------------------------------------------------------------------------------------------------------------------------------
Total                              	|  -              	|_              	|  1.6681e+04     	|  100 %          	|
--------------------------------------------------------------------------------------------------------------------------------------
run_training_epoch                 	|  8188.6         	|2              	|  1.6377e+04     	|  98.179         	|
run_training_batch                 	|  0.34554        	|46754          	|  1.6155e+04     	|  96.849         	|
optimizer_step_and_closure_0       	|  0.27064        	|46754          	|  1.2654e+04     	|  75.856         	|
training_step_and_backward         	|  0.16342        	|46754          	|  7640.6         	|  45.804         	|
model_forward                      	|  0.14814        	|46754          	|  6926.2         	|  41.521         	|
training_step                      	|  0.14799        	|46754          	|  6919.3         	|  41.48          	|
backward                           	|  0.014341       	|46754          	|  670.51         	|  4.0196         	|
on_train_batch_end                 	|  0.0010476      	|46754          	|  48.979         	|  0.29362        	|
evaluation_step_and_end            	|  13.214         	|3              	|  39.642         	|  0.23765        	|
validation_step                    	|  13.214         	|3              	|  39.641         	|  0.23764        	|
get_train_batch                    	|  0.00057629     	|46754          	|  26.944         	|  0.16152        	|
training_batch_to_device           	|  0.0003749      	|46754          	|  17.528         	|  0.10508        	|
on_after_backward                  	|  3.4193e-05     	|46754          	|  1.5987         	|  0.0095838      	|
on_batch_end                       	|  2.9265e-05     	|46754          	|  1.3683         	|  0.0082025      	|
on_batch_start                     	|  2.8412e-05     	|46754          	|  1.3284         	|  0.0079633      	|
on_before_optimizer_step           	|  2.6572e-05     	|46754          	|  1.2424         	|  0.0074478      	|
on_before_zero_grad                	|  2.6229e-05     	|46754          	|  1.2263         	|  0.0073516      	|
on_train_batch_start               	|  2.5425e-05     	|46754          	|  1.1887         	|  0.0071262      	|
training_step_end                  	|  2.2365e-05     	|46754          	|  1.0456         	|  0.0062684      	|
on_before_backward                 	|  2.2345e-05     	|46754          	|  1.0447         	|  0.0062629      	|
on_train_epoch_end                 	|  0.23711        	|2              	|  0.47421        	|  0.0028428      	|
on_train_start                     	|  0.042717       	|1              	|  0.042717       	|  0.00025608     	|
evaluation_batch_to_device         	|  0.0016145      	|3              	|  0.0048434      	|  2.9036e-05     	|
on_validation_end                  	|  0.00073581     	|3              	|  0.0022074      	|  1.3233e-05     	|
on_validation_batch_end            	|  0.00056858     	|3              	|  0.0017057      	|  1.0226e-05     	|
on_validation_start                	|  0.00040302     	|3              	|  0.001209       	|  7.248e-06      	|
on_train_epoch_start               	|  0.00042685     	|2              	|  0.0008537      	|  5.1178e-06     	|
on_validation_batch_start          	|  0.00014581     	|3              	|  0.00043743     	|  2.6223e-06     	|
on_train_end                       	|  0.00036821     	|1              	|  0.00036821     	|  2.2073e-06     	|
validation_step_end                	|  5.5551e-05     	|3              	|  0.00016665     	|  9.9905e-07     	|
on_epoch_start                     	|  2.9914e-05     	|5              	|  0.00014957     	|  8.9665e-07     	|
on_validation_epoch_end            	|  3.9617e-05     	|3              	|  0.00011885     	|  7.1249e-07     	|
on_epoch_end                       	|  1.931e-05      	|5              	|  9.6549e-05     	|  5.788e-07      	|
on_validation_epoch_start          	|  1.7585e-05     	|3              	|  5.2755e-05     	|  3.1626e-07     	|
on_fit_start                       	|  3.6187e-05     	|1              	|  3.6187e-05     	|  2.1694e-07     	|
on_train_dataloader                	|  1.6296e-05     	|1              	|  1.6296e-05     	|  9.7692e-08     	|
on_val_dataloader                  	|  1.4723e-05     	|1              	|  1.4723e-05     	|  8.8262e-08     	|
on_before_accelerator_backend_setup	|  1.044e-05      	|1              	|  1.044e-05      	|  6.2586e-08     	|


```



#### Epoch 6

08.07晚上挂机，跑了一晚上+上午到epoch6左右，iter到168000

```
100%|██████████| 1986/1986 [01:31<00:00, 21.63it/s]
2022-08-07 02:47:19.785 | DEBUG    | tools.prepare_data.generate_voxel:gen_octree:128 - number of points for voxel generation: 66163718/66163718
2022-08-07 02:47:20.088 | DEBUG    | tools.prepare_data.generate_voxel:gen_octree:152 - level: 10 for expected voxel size: 0.008139948706530686
sdf filtered points 66163718, max sdf: -0.3115154206752777, min sdf: 0.28325119614601135
Update successful!!
Epoch 6:  83%|████████▎ | 19429/23378 [2:04:58<25:24,  2.59it/s, loss=0.192, train/color_loss=0.138, train/normal_loss=0.0009, train/mask_error=0.0368, train/sfm_depth_loss=0.0113, train/psnr=21.30, val/psnr=15.00] 
```

