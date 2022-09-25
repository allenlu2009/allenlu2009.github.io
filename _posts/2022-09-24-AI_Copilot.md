---

title: AI for AI I - Copilot 
date: 2022-09-24 23:10:08
categories:
- AI
tags: [Copilot]
typora-root-url: ../../allenlu2009.github.io
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>



## Introduction



AI for AI:  use copilot for machine learning.





## Python Copilot



#### Get Max and Min of a list

非常容易，參考 reference



#### A Simple Calculator

非常容易，參考 reference



#### Plot a sine wave

只要 input '''plot asine wave using matplotlib''', 就會 generate the following code and work!!!

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_sin():
    '''
    plot a sine wave
    '''
    x = np.arange(0, 3 * np.pi, 0.1)
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    plot_sin()
```



### 



### Classification (MNIST)

是否可以用 copilot for image classification?  Yes, need more exploration.

Use Pytorch as example

use [ctl-enter] to view all examples!



Start with comment: ''' mnist example using pytorch'



First import all packages:

```python
import argparse
import time
import os
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```



再來是煉丹五部曲！

Step 1:  load dataset,  type: def load_dataset  => return mnist train_loader, test_loader!

```python
def load_dataset(path, batch_size):
    path = get_data_path(path)
    print('Loading the training dataset from: {}'.format(path))
    train_loader = DataLoader(
        datasets.MNIST(path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST(path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
```



Step 2: build a model: type : class resnet

```python
class resnet18(nn.Module):
    def __init__(self, num_classes):
        super(resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(BasicBlock(inplanes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
```



Step 3: train a network: type def train

```python
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
```



Step 4: test a trained network: type def test

```python
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```





## Julia Copilot



### Compute FFT



### Plot a sin wave

Results: failed.  Not recommend

Problem:

* Still use old Julia code not working: e.g. linspace()









## Reference