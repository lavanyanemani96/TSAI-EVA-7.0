# Session 8.0 Advanced Training Concepts

Goals:
1. To pull PyTorch_Vision repository [PyTorch_Vision](https://github.com/lavanyanemani96/PyTorch_Vision) that consists of models and other utilities. 
2. Train ResNet18 for 20 epochs
3. Show loss and accuracy curves
4. Show a gallery of 10 misclassified images
5. Show GradCam output on 10 misclassified images
6. Early submission transforms: RandomCrop(32, padding=4), CutOut(16x16)

Summary of ResNet18:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param 
================================================================
            Conv2d-1           [-1, 64, 36, 36]           1,728
       BatchNorm2d-2           [-1, 64, 36, 36]             128
            Conv2d-3           [-1, 64, 36, 36]          36,864
       BatchNorm2d-4           [-1, 64, 36, 36]             128
            Conv2d-5           [-1, 64, 36, 36]          36,864
       BatchNorm2d-6           [-1, 64, 36, 36]             128
        BasicBlock-7           [-1, 64, 36, 36]               0
            Conv2d-8           [-1, 64, 36, 36]          36,864
       BatchNorm2d-9           [-1, 64, 36, 36]             128
           Conv2d-10           [-1, 64, 36, 36]          36,864
      BatchNorm2d-11           [-1, 64, 36, 36]             128
       BasicBlock-12           [-1, 64, 36, 36]               0
           Conv2d-13          [-1, 128, 19, 19]          73,728
      BatchNorm2d-14          [-1, 128, 19, 19]             256
           Conv2d-15          [-1, 128, 19, 19]         147,456
      BatchNorm2d-16          [-1, 128, 19, 19]             256
           Conv2d-17          [-1, 128, 19, 19]           8,192
      BatchNorm2d-18          [-1, 128, 19, 19]             256
       BasicBlock-19          [-1, 128, 19, 19]               0
           Conv2d-20          [-1, 128, 19, 19]         147,456
      BatchNorm2d-21          [-1, 128, 19, 19]             256
           Conv2d-22          [-1, 128, 19, 19]         147,456
      BatchNorm2d-23          [-1, 128, 19, 19]             256
       BasicBlock-24          [-1, 128, 19, 19]               0
           Conv2d-25          [-1, 256, 11, 11]         294,912
      BatchNorm2d-26          [-1, 256, 11, 11]             512
           Conv2d-27          [-1, 256, 11, 11]         589,824
      BatchNorm2d-28          [-1, 256, 11, 11]             512
           Conv2d-29          [-1, 256, 11, 11]          32,768
      BatchNorm2d-30          [-1, 256, 11, 11]             512
       BasicBlock-31          [-1, 256, 11, 11]               0
           Conv2d-32          [-1, 256, 11, 11]         589,824
      BatchNorm2d-33          [-1, 256, 11, 11]             512
           Conv2d-34          [-1, 256, 11, 11]         589,824
      BatchNorm2d-35          [-1, 256, 11, 11]             512
       BasicBlock-36          [-1, 256, 11, 11]               0
           Conv2d-37            [-1, 512, 7, 7]       1,179,648
      BatchNorm2d-38            [-1, 512, 7, 7]           1,024
           Conv2d-39            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-40            [-1, 512, 7, 7]           1,024
           Conv2d-41            [-1, 512, 7, 7]         131,072
      BatchNorm2d-42            [-1, 512, 7, 7]           1,024
       BasicBlock-43            [-1, 512, 7, 7]               0
           Conv2d-44            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-45            [-1, 512, 7, 7]           1,024
           Conv2d-46            [-1, 512, 7, 7]       2,359,296
      BatchNorm2d-47            [-1, 512, 7, 7]           1,024
       BasicBlock-48            [-1, 512, 7, 7]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 16.96
Params size (MB): 42.63
Estimated Total Size (MB): 59.59
----------------------------------------------------------------

Training: 
EPOCH: 0
  0%|          | 0/391 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))

Loss=1.1802575588226318 Batch_id=390 Accuracy=41.11: 100%|██████████| 391/391 [03:03<00:00,  2.13it/s]

Test set: Average loss: 0.0101, Accuracy: 5390/10000 (53.90%)

EPOCH: 1
Loss=0.9105045199394226 Batch_id=390 Accuracy=59.05: 100%|██████████| 391/391 [03:04<00:00,  2.12it/s]

Test set: Average loss: 0.0081, Accuracy: 6343/10000 (63.43%)

EPOCH: 2
Loss=0.7933289408683777 Batch_id=390 Accuracy=67.36: 100%|██████████| 391/391 [03:02<00:00,  2.14it/s]

Test set: Average loss: 0.0083, Accuracy: 6489/10000 (64.89%)

EPOCH: 3
Loss=0.7567934393882751 Batch_id=390 Accuracy=72.75: 100%|██████████| 391/391 [03:03<00:00,  2.13it/s]

Test set: Average loss: 0.0058, Accuracy: 7413/10000 (74.13%)

EPOCH: 4
Loss=0.5510321259498596 Batch_id=390 Accuracy=76.38: 100%|██████████| 391/391 [03:02<00:00,  2.14it/s]

Test set: Average loss: 0.0060, Accuracy: 7341/10000 (73.41%)

EPOCH: 5
Loss=0.6900001168251038 Batch_id=390 Accuracy=78.66: 100%|██████████| 391/391 [03:03<00:00,  2.13it/s]

Test set: Average loss: 0.0068, Accuracy: 7306/10000 (73.06%)

EPOCH: 6
Loss=0.35625725984573364 Batch_id=390 Accuracy=80.73: 100%|██████████| 391/391 [03:02<00:00,  2.14it/s]

Test set: Average loss: 0.0051, Accuracy: 7893/10000 (78.93%)

EPOCH: 7
Loss=0.500881552696228 Batch_id=390 Accuracy=82.33: 100%|██████████| 391/391 [03:03<00:00,  2.13it/s]

Test set: Average loss: 0.0044, Accuracy: 8150/10000 (81.50%)

EPOCH: 8
Loss=0.28307539224624634 Batch_id=390 Accuracy=84.10: 100%|██████████| 391/391 [03:03<00:00,  2.14it/s]

Test set: Average loss: 0.0048, Accuracy: 8086/10000 (80.86%)

EPOCH: 9
Loss=0.5165377855300903 Batch_id=390 Accuracy=85.39: 100%|██████████| 391/391 [03:03<00:00,  2.13it/s]

Test set: Average loss: 0.0047, Accuracy: 8117/10000 (81.17%)

EPOCH: 10
Loss=0.35801881551742554 Batch_id=390 Accuracy=86.48: 100%|██████████| 391/391 [03:03<00:00,  2.14it/s]

Test set: Average loss: 0.0037, Accuracy: 8428/10000 (84.28%)

EPOCH: 11
Loss=0.48848551511764526 Batch_id=390 Accuracy=87.17: 100%|██████████| 391/391 [03:03<00:00,  2.13it/s]

Test set: Average loss: 0.0044, Accuracy: 8256/10000 (82.56%)

EPOCH: 12
Loss=0.3511068522930145 Batch_id=390 Accuracy=88.01: 100%|██████████| 391/391 [03:03<00:00,  2.14it/s]

Test set: Average loss: 0.0048, Accuracy: 8173/10000 (81.73%)

EPOCH: 13
Loss=0.3243023753166199 Batch_id=390 Accuracy=89.13: 100%|██████████| 391/391 [03:03<00:00,  2.14it/s]

Test set: Average loss: 0.0039, Accuracy: 8479/10000 (84.79%)

EPOCH: 14
Loss=0.2742513120174408 Batch_id=390 Accuracy=89.81: 100%|██████████| 391/391 [03:03<00:00,  2.14it/s]

Test set: Average loss: 0.0056, Accuracy: 7943/10000 (79.43%)

EPOCH: 15
Loss=0.36127573251724243 Batch_id=390 Accuracy=90.33: 100%|██████████| 391/391 [03:02<00:00,  2.14it/s]

Test set: Average loss: 0.0033, Accuracy: 8688/10000 (86.88%)

EPOCH: 16
Loss=0.16804663836956024 Batch_id=390 Accuracy=90.71: 100%|██████████| 391/391 [03:02<00:00,  2.14it/s]

Test set: Average loss: 0.0039, Accuracy: 8534/10000 (85.34%)

EPOCH: 17
Loss=0.21369759738445282 Batch_id=390 Accuracy=91.23: 100%|██████████| 391/391 [03:03<00:00,  2.13it/s]

Test set: Average loss: 0.0035, Accuracy: 8641/10000 (86.41%)

EPOCH: 18
Loss=0.22510726749897003 Batch_id=390 Accuracy=91.87: 100%|██████████| 391/391 [03:04<00:00,  2.12it/s]

Test set: Average loss: 0.0042, Accuracy: 8475/10000 (84.75%)

EPOCH: 19
Loss=0.1672838032245636 Batch_id=390 Accuracy=92.57: 100%|██████████| 391/391 [03:03<00:00,  2.13it/s]

Test set: Average loss: 0.0035, Accuracy: 8712/10000 (87.12%)


Plots: 

![Loss and Accuracy plots](./plots/loss_accuracy.png)
![Misclassified images](./plots/misclassified.png)
![GradCam output of Misclassified images](./plots/misclassified_gradcam.png)

