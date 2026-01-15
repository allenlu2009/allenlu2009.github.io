---
title: LLM Mamba Installation
date: 2025-07-20 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-1231-Perplexity]]"
categories:
  - LLM
tags:
  - Algorithm
  - GPT
  - LLM
  - Mamba
  - Perplexity
  - SSM
  - Transformer
---




## Source

Good reference 安裝 mamba:  https://zhuanlan.zhihu.com/p/25916604332

用 nvcc check cuda 版本，不要用 nvidia-smi!
我用 cuda 12.1,  python 3.10, pytorch 2.5!
 
 conda create --name mamba2 python=3.10
 conda activate mamba2
 2246  lrt
 2247  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
## Problem 1: Flash attention!

pre-built, conda 都不行。最後這版好像 ok.

	pip install flash-attn==2.5.8 --no-build-isolation

### 安裝: causal_conv1d!

使用命令查看是否支持abi
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
我的输出 false，就选择abiFALSE的
记得点show all选择对应的torch版本
pip install tools/causal_conv1d-1.5.2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

做完後 run pytest: 8794 passed, 3888 skipped in 133.40s 






From llama3 開始 source code installation

https://blog.csdn.net/yyywxk/article/details/144790950

git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.1.1 # 用你想要的版本
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
cd ..
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v1.1.1 # 用你想要的版本
pip install . # 方式一，下载whl安装，两种方式选择一个即可
MAMBA_FORCE_BUILD=TRUE pip install . # 方式二，强制在本地编译安装，Win 下无法识别此命令


Problem 3: upgrade mamba-ssm
pip install --upgrade mamba-ssm? (from 1.1 to 2.2)