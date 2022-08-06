

### exp_BG

设定参数 `IMG_DOWNSCALE=15` 以及

```
python train.py --cfg_path ${config_file} \
  --num_gpus 1 --num_nodes 1 \
  --num_epochs 2 --batch_size 1024 --test_batch_size 512 --num_workers 4 \
  --exp_name ${jobname} 2>&1|tee log/${jobname}.log \
```



#### Epoch 20

```
Epoch 19: 100%|██████████| 626/626 [06:13<00:00,  1.68it/s, loss=0.219, train/color_loss=0.169, train/normal_loss=0.00145, train/mask_error=0.0251, train/sfm_depth_loss=0.0125, train/psnr=19.90, val/psnr=16.60]
Epoch 19: 100%|██████████| 626/626 [06:13<00:00,  1.68it/s, loss=0.219, train/color_loss=0.169, train/normal_loss=0.00145, train/mask_error=0.0251, train/sfm_depth_loss=0.0125, train/psnr=19.90, val/psnr=16.60]
INFO - 2022-08-04 14:36:34,955 - base - FIT Profiler Report

Action                             	|  Mean duration (s)	|Num calls      	|  Total time (s) 	|  Percentage %   	|
--------------------------------------------------------------------------------------------------------------------------------------
Total                              	|  -              	|_              	|  7440.9         	|  100 %          	|
--------------------------------------------------------------------------------------------------------------------------------------
run_training_epoch                 	|  356.19         	|20             	|  7123.8         	|  95.738         	|
run_training_batch                 	|  0.5163         	|12500          	|  6453.7         	|  86.733         	|
optimizer_step_and_closure_0       	|  0.43823        	|12500          	|  5477.9         	|  73.619         	|
training_step_and_backward         	|  0.25746        	|12500          	|  3218.2         	|  43.251         	|
model_forward                      	|  0.2377         	|12500          	|  2971.3         	|  39.932         	|
training_step                      	|  0.23755        	|12500          	|  2969.4         	|  39.907         	|
evaluation_step_and_end            	|  28.295         	|21             	|  594.19         	|  7.9855         	|
validation_step                    	|  28.295         	|21             	|  594.19         	|  7.9854         	|
backward                           	|  0.018791       	|12500          	|  234.89         	|  3.1567         	|
get_train_batch                    	|  0.0017107      	|12500          	|  21.383         	|  0.28737        	|
on_train_batch_end                 	|  0.0010557      	|12500          	|  13.196         	|  0.17735        	|
training_batch_to_device           	|  0.00038316     	|12500          	|  4.7895         	|  0.064367       	|
on_train_epoch_end                 	|  0.22528        	|20             	|  4.5055         	|  0.060551       	|
on_after_backward                  	|  3.3122e-05     	|12500          	|  0.41403        	|  0.0055643      	|
on_batch_start                     	|  3.0598e-05     	|12500          	|  0.38248        	|  0.0051402      	|
on_batch_end                       	|  3.0303e-05     	|12500          	|  0.37879        	|  0.0050906      	|
on_before_optimizer_step           	|  2.9181e-05     	|12500          	|  0.36476        	|  0.0049021      	|
on_train_batch_start               	|  2.7165e-05     	|12500          	|  0.33956        	|  0.0045634      	|
on_before_zero_grad                	|  2.7151e-05     	|12500          	|  0.33939        	|  0.0045611      	|
on_before_backward                 	|  2.4814e-05     	|12500          	|  0.31017        	|  0.0041685      	|
training_step_end                  	|  2.0135e-05     	|12500          	|  0.25169        	|  0.0033825      	|
evaluation_batch_to_device         	|  0.0014027      	|21             	|  0.029456       	|  0.00039587     	|
on_validation_end                  	|  0.00086022     	|21             	|  0.018065       	|  0.00024277     	|
on_train_start                     	|  0.015743       	|1              	|  0.015743       	|  0.00021157     	|
on_validation_batch_end            	|  0.00055351     	|21             	|  0.011624       	|  0.00015621     	|
on_validation_start                	|  0.00051907     	|21             	|  0.0109         	|  0.00014649     	|
on_train_epoch_start               	|  0.0005138      	|20             	|  0.010276       	|  0.0001381      	|
on_validation_batch_start          	|  0.0001437      	|21             	|  0.0030177      	|  4.0555e-05     	|
on_epoch_start                     	|  2.8559e-05     	|41             	|  0.0011709      	|  1.5736e-05     	|
on_epoch_end                       	|  2.4445e-05     	|41             	|  0.0010022      	|  1.3469e-05     	|
validation_step_end                	|  4.4674e-05     	|21             	|  0.00093815     	|  1.2608e-05     	|
on_validation_epoch_end            	|  4.1283e-05     	|21             	|  0.00086694     	|  1.1651e-05     	|
on_validation_epoch_start          	|  2.2263e-05     	|21             	|  0.00046753     	|  6.2832e-06     	|
on_train_end                       	|  0.00036496     	|1              	|  0.00036496     	|  4.9048e-06     	|
on_fit_start                       	|  3.4309e-05     	|1              	|  3.4309e-05     	|  4.6109e-07     	|
on_val_dataloader                  	|  1.9739e-05     	|1              	|  1.9739e-05     	|  2.6528e-07     	|
on_train_dataloader                	|  1.5682e-05     	|1              	|  1.5682e-05     	|  2.1075e-07     	|
on_before_accelerator_backend_setup	|  1.0413e-05     	|1              	|  1.0413e-05     	|  1.3994e-07     	|
```



#### Epoch 50

```
start marching cubes
radius:  0.314929498991263

Epoch 49: 100%|██████████| 626/626 [03:57<00:00,  2.64it/s, loss=0.182, train/color_loss=0.145, train/normal_loss=0.00192, train/mask_error=0.0242, train/sfm_depth_loss=0.00774, train/psnr=20.80, val/psnr=16.60]
Epoch 49: 100%|██████████| 626/626 [03:57<00:00,  2.64it/s, loss=0.182, train/color_loss=0.145, train/normal_loss=0.00192, train/mask_error=0.0242, train/sfm_depth_loss=0.00774, train/psnr=20.80, val/psnr=16.60]
INFO - 2022-08-06 08:24:36,190 - base - FIT Profiler Report

Action                             	|  Mean duration (s)	|Num calls      	|  Total time (s) 	|  Percentage %   	|
--------------------------------------------------------------------------------------------------------------------------------------
Total                              	|  -              	|_              	|  9027.7         	|  100 %          	|
--------------------------------------------------------------------------------------------------------------------------------------
run_training_epoch                 	|  238.11         	|33             	|  7857.5         	|  87.037         	|
run_training_batch                 	|  0.34761        	|20625          	|  7169.5         	|  79.416         	|
optimizer_step_and_closure_0       	|  0.33665        	|20625          	|  6943.3         	|  76.911         	|
training_step_and_backward         	|  0.16322        	|20625          	|  3366.5         	|  37.29          	|
model_forward                      	|  0.14802        	|20625          	|  3052.9         	|  33.817         	|
training_step                      	|  0.14788        	|20625          	|  3049.9         	|  33.784         	|
evaluation_step_and_end            	|  16.504         	|34             	|  561.14         	|  6.2157         	|
validation_step                    	|  16.504         	|34             	|  561.13         	|  6.2156         	|
backward                           	|  0.014307       	|20625          	|  295.08         	|  3.2686         	|
get_train_batch                    	|  0.0019783      	|20625          	|  40.803         	|  0.45197        	|
on_train_batch_end                 	|  0.0010643      	|20625          	|  21.952         	|  0.24316        	|
on_train_epoch_end                 	|  0.25346        	|33             	|  8.3641         	|  0.092649       	|
training_batch_to_device           	|  0.00037579     	|20625          	|  7.7507         	|  0.085854       	|
on_after_backward                  	|  3.1885e-05     	|20625          	|  0.65762        	|  0.0072845      	|
on_batch_end                       	|  3.0825e-05     	|20625          	|  0.63576        	|  0.0070423      	|
on_batch_start                     	|  2.8987e-05     	|20625          	|  0.59785        	|  0.0066224      	|
on_before_optimizer_step           	|  2.6962e-05     	|20625          	|  0.5561         	|  0.0061599      	|
on_before_zero_grad                	|  2.6468e-05     	|20625          	|  0.5459         	|  0.0060469      	|
on_train_batch_start               	|  2.5408e-05     	|20625          	|  0.52404        	|  0.0058047      	|
on_before_backward                 	|  2.2562e-05     	|20625          	|  0.46533        	|  0.0051545      	|
training_step_end                  	|  2.172e-05      	|20625          	|  0.44798        	|  0.0049622      	|
evaluation_batch_to_device         	|  0.001221       	|34             	|  0.041514       	|  0.00045985     	|
on_train_start                     	|  0.039938       	|1              	|  0.039938       	|  0.00044239     	|
on_validation_end                  	|  0.0008772      	|34             	|  0.029825       	|  0.00033037     	|
on_validation_batch_end            	|  0.00055475     	|34             	|  0.018862       	|  0.00020893     	|
on_train_epoch_start               	|  0.00052435     	|33             	|  0.017304       	|  0.00019167     	|
on_validation_start                	|  0.0004564      	|34             	|  0.015518       	|  0.00017189     	|
on_validation_batch_start          	|  0.00013573     	|34             	|  0.0046149      	|  5.1119e-05     	|
validation_step_end                	|  5.4308e-05     	|34             	|  0.0018465      	|  2.0453e-05     	|
on_epoch_start                     	|  2.7151e-05     	|67             	|  0.0018191      	|  2.015e-05      	|
on_epoch_end                       	|  2.2962e-05     	|67             	|  0.0015384      	|  1.7041e-05     	|
on_validation_epoch_end            	|  3.9157e-05     	|34             	|  0.0013313      	|  1.4747e-05     	|
on_validation_epoch_start          	|  2.4476e-05     	|34             	|  0.00083217     	|  9.218e-06      	|
on_train_end                       	|  0.00038386     	|1              	|  0.00038386     	|  4.252e-06      	|
on_fit_start                       	|  4.5504e-05     	|1              	|  4.5504e-05     	|  5.0405e-07     	|
on_train_dataloader                	|  1.617e-05      	|1              	|  1.617e-05      	|  1.7911e-07     	|
on_val_dataloader                  	|  1.5067e-05     	|1              	|  1.5067e-05     	|  1.669e-07      	|
on_before_accelerator_backend_setup	|  1.2071e-05     	|1              	|  1.2071e-05     	|  1.3371e-07     	|
```

