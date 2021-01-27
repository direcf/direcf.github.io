---
title: "AlexNet, VGG, GoogLeNet, ResNet"
excerpt: "Understanding the entire system"
permalink: /docs/page1001/
author_profile: true
layout: single
classes: wide
comments: true
header:
    image: /assets/images/header2.jpg
---
## <span style="color:#3498DB">Assignment1.</span> Understanding the entire system
**1. import module**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# tensorboard
from tensorboardX import SummaryWriter
# matrix calculation
import numpy as np
import random
# learning visualization
from tqdm import tqdm
# access to file system
import os
# network composition
from network import VGG
from network import ResNet
```

**2. Dataloader**
```python
#Apply data augmentation and data preprocessing for training set
transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # Random Crop: Randomly crop the part of the large image and utilize it as an augmented data 
        transforms.RandomHorizontalFlip(), # Random Horizontal Flip: Randomly flip the image and utilize it as an augmented data
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023,0.1994,0.2010]), # Normalize the data using the given mean and standard deviation
        ])

#Apply data preprocessing for test set
transform_test = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023,0.1994,0.2010]),
        ]) 

# torchvision dataset : MNIST, Fashion-MNIST, KMNIST, EMNIST, FakeData, COCO, LSUN, ImageFolder, DatasetFolder, Imagenet-12, CIFAR, STL10, SVHN, SBU, Flickr, VOC, Cityscapes
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=False)
```

**3. Optimizer, Loss function**
```python
def train(model, n_epoch, loader, optimizer, criterion, device="cpu"):
  model.train() # train mode
  for epoch in tqdm(range(n_epoch)):
    running_loss = 0.0
    # Usage for Dataloader
    for i, data in enumerate(loader, 0): 
      images, labels = data
      # device = "cpu" or "cuda"
      images = images.to(device) 
      labels = labels.to(device)
      optimizer.zero_grad() # initialize gradient
      outputs = model(images)
      loss = criterion(input=outputs, target=labels) # define loss function
      loss.backward() # backpropagation
      optimizer.step() # update weight, bias
      running_loss += loss.item()
    print('Epoch {}, loss = {:.3f}'.format(epoch, running_loss/len(loader)))
  print('Training Finished')

def evaluate(model, loader, device="cpu"):
  model.eval() # eval mode
  total=0
  correct=0
  with torch.no_grad(): # No update weight, bias
    for data in loader:
      images, labels = data
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted==labels).sum().item()
  acc = 100*correct/total
  return acc
```

**4. Network**

**<span style="color:#4287f5">(1) VGG</span>**

<a href="https://imgur.com/Sb5NTGr"><img src="https://i.imgur.com/Sb5NTGr.png" title="source: imgur.com" /></a>



**<span style="color:#4287f5">(2) ResNet</span>**

<a href="https://imgur.com/Uvj6Lu6"><img src="https://i.imgur.com/Uvj6Lu6.png" title="source: imgur.com" /></a>

**5. main function**
```python
def reset_seed(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

# VGG main function
reset_seed(0)
vgg_model = VGG().to("cuda")
criterion = nn.CrossEntropyLoss()
tb_log = SummaryWriter(log_dir=os.path.join('./', 'tensorboard'))
optimizer = optim.SGD(params=vgg_model.parameters(), lr=0.1, momentum=0.9)
train(model=vgg_model, n_epoch=10, loader=train_loader, optimizer=optimizer, criterion=criterion, device="cuda")
vgg_acc = evaluate(vgg_model, test_loader, device="cuda")
print('VGG Test accuracy: {:.2f}%'.format(vgg_acc))

# ResNet main function
reset_seed(0)
resnet_model = ResNet().to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=resnet_model.parameters(), lr=0.1, momentum=0.9)
train(model=resnet_model, n_epoch=10, loader=train_loader, optimizer=optimizer, criterion=criterion, device="cuda")
resnet_acc = evaluate(resnet_model, test_loader, device="cuda")
print('ResNet Test accuracy: {:.2f}%'.format(resnet_acc))

```

**6. tensorboard, argparse, mgpus, logging, lr_scheduler**
