---

title: AI for AI (I) - Copilot I
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



AI for AI:  use github Copilot for general signal processing, plotting, and machine learning.





## Python Copilot

結論：Copilot 對一般的 python programming 初步看起來不錯。對於 machine learning 部分還要再測試。 



#### Get Max and Min of a list

非常容易，參考 reference



#### A Simple Calculator

非常容易，參考 reference



#### Plot a sine wave

只要 type '''plot a sine wave using matplotlib''', 就會 generate the following code and work!!!

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

<img src="/media/image-20220925194637375.png" alt="image-20220925194637375" style="zoom:50%;" />

#### Compute a FFT of a sine wave and plot

Type '''compute fft of a signal and plot it''', 就會得到以下的 FFT 以及 plot in linear or log scale!

```
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

x = np.linspace(0, 1, 1000, False)
y = np.sin(50 * 2 * np.pi * x) + 0.5 * np.sin(80 * 2 * np.pi * x)

yf = fft(y)
xf = fftfreq(len(y), 1 / 1000)

# plot in linear scale
plt.plot(xf, np.abs(yf))
# plot in log scale
# plt.plot(xf, 20 * np.log10(np.abs(yf)))
# label the axes
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
# set the title
plt.title('FFT of sine waves')
# set the axis limits
plt.xlim(-100, 100)
plt.ylim(-10, 550)
# display the plot
plt.grid()
plt.show()
```



<img src="/media/image-20220925194507181.png" alt="image-20220925194507181" style="zoom:67%;" />



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

用幾個常見的例子。

### Plot a sin wave

Type "plot a sine wave"

```julia
using PyPlot

x = linspace(0, 2pi, 1000)
y = sin(x)

PyPlot.plot(x, y, color="red", linewidth=2.0, linestyle="-")
xlabel("x")
ylabel("sin(x)")
title("sin(x)")
PyPlot.grid("on")

PyPlot.savefig("plotSin.png")
```

Results: failed.  

Problem and fix.

* Still use old Julia code not working: linspace(0, 2pi, 1000) -> range(0, 2pi; length=1000) 
* No vector operation:  y = sin(x) -> y = sin.(x)
* Figure not display!  Add display(gcf())

修改的版本 and work.

```julia
using PyPlot

x = range(0, 2pi; length = 1000)
y = sin.(x)

figure()
PyPlot.plot(x, y, color="red", linewidth=2.0, linestyle="-")
xlabel("x")
ylabel("sin(x)")
title("sin(x)")
PyPlot.grid("on")
display(gcf())

PyPlot.savefig("plotSin.png")
```

<img src="/media/image-20220925201751196.png" alt="image-20220925201751196" style="zoom:50%;" />

不過我再 type:  plot a cosine wave,  Copilot 可以現學現賣！

```julia
# plot a cosine wave

using PyPlot

x = range(0, 2pi; length = 1000)
y = cos.(x)

figure()
PyPlot.plot(x, y, color="blue", linewidth=2.0, linestyle="-")
xlabel("x")
ylabel("cos(x)")
title("cos(x)")
PyPlot.grid("on")

display(gcf())
```

<img src="/media/image-20220925201939755.png" alt="image-20220925201939755" style="zoom:50%;" />

### Compute and Plot FFT

再 type "compute the FFT of a signal and plto the result".  還是不行！

```julia
using PyPlot

# define the signal
t = linspace(0, 1, 1000)
x = sin(2*pi*10*t) + sin(2*pi*20*t)

# compute the FFT
X = fft(x)

# plot the result
figure()
subplot(2,1,1)
plot(t, x)
title("Signal")
subplot(2,1,2)
plot(abs(X))
title("FFT")
```

* Problem and fix.

  * linspace(0, 2pi, 1000) -> range(0, 2pi; length=1000) 
  * No vector operation:  sin(x) -> sin.(x);  abs(x) -> abs.(x)
  * plot -> PyPlot.plot
  * Figure not display!  Add display(gcf())

  修改的版本 and work.

```julia
using PyPlot
using FFTW

# define the signal
t = range(0, 1; length = 1000)
x = sin.(2*pi*10*t) + sin.(2*pi*20*t)

# compute the FFT
X = fft(x)

# plot the result
figure()
subplot(2,1,1)
PyPlot.plot(t, x)
title("Signal")
subplot(2,1,2)
PyPlot.plot(abs.(X))
title("FFT")
display(gcf())
```

<img src="/media/image-20220925203225745.png" alt="image-20220925203225745" style="zoom:50%;" />





### Compute and Plot Self Entropy

基本上 input title 和 input signal range.  **Copilot 自動 show 出 entropy 的 formula in vector form!**

```
# compute the entropy of a normal distribution and plot the result

using PyPlot
#using FFTW

# define the signal
σ = range(0.01, 10; step = 0.01)

# compute the entropy
H = 0.5 * log.(2 * π * exp.(1) * σ.^2)

# plot the result
figure()
PyPlot.plot(σ, H, color="blue", linewidth=2.0, linestyle="-")
xlim(-0.1, 10)
ylim(-4, 4)
xlabel("σ")
ylabel("Entropy")
#title("Entropy")
PyPlot.grid("on")
display(gcf())
```

<img src="/media/image-20221001081647206.png" alt="image-20221001081647206" style="zoom:50%;" />

## Reference