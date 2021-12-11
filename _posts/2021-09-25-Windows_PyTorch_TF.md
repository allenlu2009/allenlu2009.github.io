---
title: Windows + CUDA ML - PyTorch and TensorFlow
categories: 
- AI
tags: [Windows, terminal, machine learning, ML, GPU, CUDA, PowerShell]
typora-root-url: ../../allenlu2009.github.io
---

最正宗安裝和執行 PyTorch and TensorFlow 都是在 Ubuntu platform, 特別是在有 GPU 的情況下。不過最近開始有一些在 Windows 10 執行 PyTorch and TensorFlow 需求。
* 容易同時 blogging 和 coding, 因爲 blogging tool 大多是在 Windows (or Mac) 環境下，例如 Typora, Jekyll, etc.
* Windows system 開始有一些令人驚喜的進步，for example,
  * Windows 10 開始支持 Windows subsystem for Linux (WSL2).   Windows 11 將會支持 Windows subsystem for Android (WSA).  WSL2 甚至開始支持 GPU.
  * Windows 的 PowerShell and Terminal 越來越像 Linux shell.  可以做一些 coding. 
* 另一個之後會比較 Windows native machine learning platform, i.e. DirectML 的效能和 Pytorch/Tensorflow on Windows 的比較。



### Windows Installation

Windows 安裝和 Ubuntu 基本相同。我的習慣是接近硬體的開始安裝，順序會是GPU driver→CUDA Toolkit→cuDNN→Anaconda/Python→Pytorch and Tensorflow。

Anaconda 的好處是可以建立不同的環境，安裝不同版本的 Tensorflow (1.15, 2.x) 和 PyTorch. 

這裏用

* GTX 1080: GPU driver 456.71
* CUDA: 10.2
* CuDNN: 8.2.2
* Anaconda 2020.11: 2.0.4 (Python 3.8)
* PyTorch: 1.9.1  (1 conda env: torch)
* TensorFlow: 1.15 and 2.3 (2 conda env, Python down to 3.7 for compatibility)



#### Step 1:  GPU driver

下表示 CUDA 對應 driver table.   我的 driver version 符合 CUDA 10.2 需求。因此不用再 update driver.

<img src="/media/image-20210925235445225.png" alt="image-20210925235445225" style="zoom:100%;" />

#### Step 2:  Install CUDA and CuDNN

CUDA 是 Nvidia 的 (GPU) computing platform API;  CuDNN 是特別針對 machine learning 的加强 API.

CUDA download: https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork 可以直接 install.

Validate:  Use PowerShell  PS> nvcc -V



Install CuDNN 麻煩一點。必須在 Nvidia 官網注冊。Download and decompress 后，再把 bin, include, lib copy 到 CUDA 對應的 directory.    

  <img src="/media/image-20210926000950554.png" alt="image-20210926000950554" style="zoom:80%;" />

<img src="/media/image-20210926001104446.png" alt="image-20210926001104446" style="zoom:80%;" />

Check the environment variable of PATH, and CUDA, etc.



#### Step 3:  Install Anaconda

直接在官網 download and install Anaconda.  我的是 2020.11 version: 2.0.4 (default Python 3.8)

接著用 conda create 3 個 virtual environments: 

* torch for PyTorch 1.9
* tf1 for TensorFlow-GPU 1.15
* tf2 for TensorFlow-GPU 2.6

First, create torch.  之所以 clone base 因爲 base 已經包含 matplotlib, scipy, etc. 常用 packages.
```
conda create --name torch --clone base  (base Python is 3.8)
conda activate torch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
Next, create tf2.  因爲 TensforFlow-GPU 和 Python version 以及 CUDA 強相關。我試了幾種 configurations.  TensorFlow 2.3 和 Python 3.7 以及 CUDA 10.2 看來最合。
```
conda activate base
conda create --name tf2 python=3.7  
conda activate tf2
conda install tensorflow-gpu=2.3

conda install -c conda-forge matplotlib
```
Next, create tf1.  爲什麽還要 tensorflow1?  因爲很多 legacy code 仍然使用 tensorflow1. 我試了幾種 configurations.  TensorFlow 1.15 和 Python 3.6 以及 CUDA 10.0 看來最合。
```
conda activate base
conda create --name tf1 python=3.6  
conda activate tf1
conda install tensorflow-gpu=1.15

conda install -c conda-forge matplotlib
```



Keras is built-in in tensorflow 2.x.  但在 tensorflow 1.x 需要額外 install.  Use anaconda to install Keras!

Jupyter notebook 也需要額外 install!



Test using mnist



#### Step 4:  Update Windows Terminal Setting

Windows 10 Terminal 有一些優點

* With multi-tab support

* Support Linux command : wsl <linux command>
* Shell ?



Add anaconda3 PowerShell to Terminal list.

Ref: https://zhuanlan.zhihu.com/p/364292785

 

In the setting.json, add the following anaconda prompt:

 

{

​        // Anaconda Prompt

​        "guid": "{2daaf818-fbab-47e8-b8ba-2f82eb89de40}",

​        "name": "Anaconda Prompt",

​        "icon": "D:\\anaconda3\\Menu\\anaconda-navigator.ico",

​        "startingDirectory":"%USERPROFILE%\\OneDrive\\GitHub",

​        "commandline": "powershell.exe -ExecutionPolicy ByPass -NoExit -Command \"& 'D:\\anaconda3\\shell\\condabin\\conda-hook.ps1' ; conda activate 'D:\\anaconda3' \" ",

​        "hidden": false

​      }

 

#### Step 5:  Clone GitHub Coding to Local OneDrive/GitHub



#### Step 6:  Install VS Code As Coding Tool

