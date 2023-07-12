### Writing a custom resnet architecture for CIFAR10
![cr](./images/param.png)

### Applying data transforms for train data set
- Padding of 4 <br/>
- RandomCrop 32, 32 <br/>
- FlipLR (Horizontal flip)  <br/>
- CutOut(8, 8)  <br/>
![dt](./images/train_images.png)

### Hyperparameters
- Number of epochs - 24
- Batch size - 512
- Optimizer - Adam
- Scheduler - One Cycle Policy
  - Max learning rate at 5th Epoch <br/>
  Set pct_start of [pytorch's OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) to 0.2 (5/24 epoch) <br/>
  ![lr](./images/learning_rate_plot.png)
  - Maximum learning rate set to **0.0008111308307896874**. This learning rate is calculated by torch-lr-finder package from [pytorch-lr-finder](https://github.com/davidtvs/pytorch-lr-finder)
  ![max learning rate](./images/learning_rate.png)

### Losses and accuracy
- Maximum Training accuracy - 96.86%
- Maximum test accuracy - 90.90%
![plots](./images/loss_acc_plot.png)

### Training logs
![logs](./images/logs.png)
