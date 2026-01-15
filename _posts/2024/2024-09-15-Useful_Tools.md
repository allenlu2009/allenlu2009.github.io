






## Vs Code Python and Colab Jupyter Notebook Co-work

Vs Code Python and Colab Jupyter Notebook have their own advantages.    

Vs Code Python:  
1. very good debugging environment including breakpoint, watch function.
2. use local GPU card to save money

Jupyter
1. Python file compatible
2. Use remote GPU with better performance
3. Interactive debugging without recompile

Here's a comparison table summarizing the features of VS Code and Jupyter for Python development:

| Feature               | VS Code                                                  | Jupyter                                     |
| --------------------- | -------------------------------------------------------- | ------------------------------------------- |
| Debugging Environment | Very good debugging with breakpoints and watch functions | Interactive debugging without recompilation |
| GPU Utilization       | Uses local GPU card to save money                        | Uses remote GPU with better performance     |
| File Compatibility    | Supports Python files                                    | Compatible with Python files                |
| Interactivity         | Less interactive, primarily code-based                   | Highly interactive with notebooks           |
| File comparison       | Easy for TEXT file, diff or BeyongCompare                | JSON file, diffcult to compare              |

### How to Avoid Two Copy of Files?

Apparently we don't really want to keep track two copies of files,  VS Code python and Colab Jupyter Notebook (ipynb).   However, there are two different file formats.   

A good way is to make the python code to be function only for importing.   The test code can be either using  `if __name__ == "__main__":`  code block, or use test folder.   This is a standard procedure so that I won't explain it.

On the other hand, the Colab Jupyter notebook needs some work.

#### Mount Local Folder
```python
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("Running in Google Colab")
    from google.colab import drive
    drive.mount('/content/drive')
    root_path = "./drive/MyDrive/ml_code/Cursor/nanogpt/"
    model_directory = "./drive/MyDrive/ml_code/model/"
    data_directory = "./drive/MyDrive/ml_code/Cursor/nanogpt/shakespeare"
else:
    print("Not running in Google Colab")
    root_path = "./"
    model_path = "/mnt/c/Users/allen/llama/llama-2-7b-chat-bin"

data_directory = "./shakespeare"
```

#### Change Directory for Import

```
%pwd
%cd {root_path}
%ls
```


#### Import related python and packages

The first line is the most important part!
```python
from nanogpt2 import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import os
from tqdm import tqdm
import argparse
```


#### Command Line Argement

```python
#args = argparse.Namespace(generate="Before we proceed any further")
args = argparse.Namespace(generate=False)
```




## Visualize transformer tensor shape and validate

### A very good tools is summary

```python
from torchinfo import summary
summary(decoder,input_size=(batch_size, config.block_size),dtypes=[torch.long],depth=5,)


====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
ShakeGPT                                           [32, 256, 65]             --
├─Embedding: 1-1                                   [32, 256, 256]            16,640
├─Embedding: 1-2                                   [256, 256]                65,536
├─Sequential: 1-3                                  [32, 256, 256]            --
│    └─TransformerBlock: 2-1                       [32, 256, 256]            --
│    │    └─LayerNorm: 3-1                         [32, 256, 256]            512
│    │    └─MultiHeadAttention: 3-2                [32, 256, 256]            --
│    │    │    └─ModuleList: 4-1                   --                        --
│    │    │    │    └─AttentionHead: 5-1           [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-2           [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-3           [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-4           [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-5           [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-6           [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-7           [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-8           [32, 256, 32]             24,576
│    │    │    └─Linear: 4-2                       [32, 256, 256]            65,792
│    │    │    └─Dropout: 4-3                      [32, 256, 256]            --
│    │    └─LayerNorm: 3-3                         [32, 256, 256]            512
│    │    └─FeedForward: 3-4                       [32, 256, 256]            --
│    │    │    └─Sequential: 4-4                   [32, 256, 256]            --
│    │    │    │    └─Linear: 5-9                  [32, 256, 1024]           263,168
│    │    │    │    └─ReLU: 5-10                   [32, 256, 1024]           --
│    │    │    │    └─Linear: 5-11                 [32, 256, 256]            262,400
│    │    │    │    └─Dropout: 5-12                [32, 256, 256]            --
│    └─TransformerBlock: 2-2                       [32, 256, 256]            --
│    │    └─LayerNorm: 3-5                         [32, 256, 256]            512
│    │    └─MultiHeadAttention: 3-6                [32, 256, 256]            --
│    │    │    └─ModuleList: 4-5                   --                        --
│    │    │    │    └─AttentionHead: 5-13          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-14          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-15          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-16          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-17          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-18          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-19          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-20          [32, 256, 32]             24,576
│    │    │    └─Linear: 4-6                       [32, 256, 256]            65,792
│    │    │    └─Dropout: 4-7                      [32, 256, 256]            --
│    │    └─LayerNorm: 3-7                         [32, 256, 256]            512
│    │    └─FeedForward: 3-8                       [32, 256, 256]            --
│    │    │    └─Sequential: 4-8                   [32, 256, 256]            --
│    │    │    │    └─Linear: 5-21                 [32, 256, 1024]           263,168
│    │    │    │    └─ReLU: 5-22                   [32, 256, 1024]           --
│    │    │    │    └─Linear: 5-23                 [32, 256, 256]            262,400
│    │    │    │    └─Dropout: 5-24                [32, 256, 256]            --
│    └─TransformerBlock: 2-3                       [32, 256, 256]            --
│    │    └─LayerNorm: 3-9                         [32, 256, 256]            512
│    │    └─MultiHeadAttention: 3-10               [32, 256, 256]            --
│    │    │    └─ModuleList: 4-9                   --                        --
│    │    │    │    └─AttentionHead: 5-25          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-26          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-27          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-28          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-29          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-30          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-31          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-32          [32, 256, 32]             24,576
│    │    │    └─Linear: 4-10                      [32, 256, 256]            65,792
│    │    │    └─Dropout: 4-11                     [32, 256, 256]            --
│    │    └─LayerNorm: 3-11                        [32, 256, 256]            512
│    │    └─FeedForward: 3-12                      [32, 256, 256]            --
│    │    │    └─Sequential: 4-12                  [32, 256, 256]            --
│    │    │    │    └─Linear: 5-33                 [32, 256, 1024]           263,168
│    │    │    │    └─ReLU: 5-34                   [32, 256, 1024]           --
│    │    │    │    └─Linear: 5-35                 [32, 256, 256]            262,400
│    │    │    │    └─Dropout: 5-36                [32, 256, 256]            --
│    └─TransformerBlock: 2-4                       [32, 256, 256]            --
│    │    └─LayerNorm: 3-13                        [32, 256, 256]            512
│    │    └─MultiHeadAttention: 3-14               [32, 256, 256]            --
│    │    │    └─ModuleList: 4-13                  --                        --
│    │    │    │    └─AttentionHead: 5-37          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-38          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-39          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-40          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-41          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-42          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-43          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-44          [32, 256, 32]             24,576
│    │    │    └─Linear: 4-14                      [32, 256, 256]            65,792
│    │    │    └─Dropout: 4-15                     [32, 256, 256]            --
│    │    └─LayerNorm: 3-15                        [32, 256, 256]            512
│    │    └─FeedForward: 3-16                      [32, 256, 256]            --
│    │    │    └─Sequential: 4-16                  [32, 256, 256]            --
│    │    │    │    └─Linear: 5-45                 [32, 256, 1024]           263,168
│    │    │    │    └─ReLU: 5-46                   [32, 256, 1024]           --
│    │    │    │    └─Linear: 5-47                 [32, 256, 256]            262,400
│    │    │    │    └─Dropout: 5-48                [32, 256, 256]            --
│    └─TransformerBlock: 2-5                       [32, 256, 256]            --
│    │    └─LayerNorm: 3-17                        [32, 256, 256]            512
│    │    └─MultiHeadAttention: 3-18               [32, 256, 256]            --
│    │    │    └─ModuleList: 4-17                  --                        --
│    │    │    │    └─AttentionHead: 5-49          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-50          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-51          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-52          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-53          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-54          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-55          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-56          [32, 256, 32]             24,576
│    │    │    └─Linear: 4-18                      [32, 256, 256]            65,792
│    │    │    └─Dropout: 4-19                     [32, 256, 256]            --
│    │    └─LayerNorm: 3-19                        [32, 256, 256]            512
│    │    └─FeedForward: 3-20                      [32, 256, 256]            --
│    │    │    └─Sequential: 4-20                  [32, 256, 256]            --
│    │    │    │    └─Linear: 5-57                 [32, 256, 1024]           263,168
│    │    │    │    └─ReLU: 5-58                   [32, 256, 1024]           --
│    │    │    │    └─Linear: 5-59                 [32, 256, 256]            262,400
│    │    │    │    └─Dropout: 5-60                [32, 256, 256]            --
│    └─TransformerBlock: 2-6                       [32, 256, 256]            --
│    │    └─LayerNorm: 3-21                        [32, 256, 256]            512
│    │    └─MultiHeadAttention: 3-22               [32, 256, 256]            --
│    │    │    └─ModuleList: 4-21                  --                        --
│    │    │    │    └─AttentionHead: 5-61          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-62          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-63          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-64          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-65          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-66          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-67          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-68          [32, 256, 32]             24,576
│    │    │    └─Linear: 4-22                      [32, 256, 256]            65,792
│    │    │    └─Dropout: 4-23                     [32, 256, 256]            --
│    │    └─LayerNorm: 3-23                        [32, 256, 256]            512
│    │    └─FeedForward: 3-24                      [32, 256, 256]            --
│    │    │    └─Sequential: 4-24                  [32, 256, 256]            --
│    │    │    │    └─Linear: 5-69                 [32, 256, 1024]           263,168
│    │    │    │    └─ReLU: 5-70                   [32, 256, 1024]           --
│    │    │    │    └─Linear: 5-71                 [32, 256, 256]            262,400
│    │    │    │    └─Dropout: 5-72                [32, 256, 256]            --
│    └─TransformerBlock: 2-7                       [32, 256, 256]            --
│    │    └─LayerNorm: 3-25                        [32, 256, 256]            512
│    │    └─MultiHeadAttention: 3-26               [32, 256, 256]            --
│    │    │    └─ModuleList: 4-25                  --                        --
│    │    │    │    └─AttentionHead: 5-73          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-74          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-75          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-76          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-77          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-78          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-79          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-80          [32, 256, 32]             24,576
│    │    │    └─Linear: 4-26                      [32, 256, 256]            65,792
│    │    │    └─Dropout: 4-27                     [32, 256, 256]            --
│    │    └─LayerNorm: 3-27                        [32, 256, 256]            512
│    │    └─FeedForward: 3-28                      [32, 256, 256]            --
│    │    │    └─Sequential: 4-28                  [32, 256, 256]            --
│    │    │    │    └─Linear: 5-81                 [32, 256, 1024]           263,168
│    │    │    │    └─ReLU: 5-82                   [32, 256, 1024]           --
│    │    │    │    └─Linear: 5-83                 [32, 256, 256]            262,400
│    │    │    │    └─Dropout: 5-84                [32, 256, 256]            --
│    └─TransformerBlock: 2-8                       [32, 256, 256]            --
│    │    └─LayerNorm: 3-29                        [32, 256, 256]            512
│    │    └─MultiHeadAttention: 3-30               [32, 256, 256]            --
│    │    │    └─ModuleList: 4-29                  --                        --
│    │    │    │    └─AttentionHead: 5-85          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-86          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-87          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-88          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-89          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-90          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-91          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-92          [32, 256, 32]             24,576
│    │    │    └─Linear: 4-30                      [32, 256, 256]            65,792
│    │    │    └─Dropout: 4-31                     [32, 256, 256]            --
│    │    └─LayerNorm: 3-31                        [32, 256, 256]            512
│    │    └─FeedForward: 3-32                      [32, 256, 256]            --
│    │    │    └─Sequential: 4-32                  [32, 256, 256]            --
│    │    │    │    └─Linear: 5-93                 [32, 256, 1024]           263,168
│    │    │    │    └─ReLU: 5-94                   [32, 256, 1024]           --
│    │    │    │    └─Linear: 5-95                 [32, 256, 256]            262,400
│    │    │    │    └─Dropout: 5-96                [32, 256, 256]            --
│    └─TransformerBlock: 2-9                       [32, 256, 256]            --
│    │    └─LayerNorm: 3-33                        [32, 256, 256]            512
│    │    └─MultiHeadAttention: 3-34               [32, 256, 256]            --
│    │    │    └─ModuleList: 4-33                  --                        --
│    │    │    │    └─AttentionHead: 5-97          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-98          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-99          [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-100         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-101         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-102         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-103         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-104         [32, 256, 32]             24,576
│    │    │    └─Linear: 4-34                      [32, 256, 256]            65,792
│    │    │    └─Dropout: 4-35                     [32, 256, 256]            --
│    │    └─LayerNorm: 3-35                        [32, 256, 256]            512
│    │    └─FeedForward: 3-36                      [32, 256, 256]            --
│    │    │    └─Sequential: 4-36                  [32, 256, 256]            --
│    │    │    │    └─Linear: 5-105                [32, 256, 1024]           263,168
│    │    │    │    └─ReLU: 5-106                  [32, 256, 1024]           --
│    │    │    │    └─Linear: 5-107                [32, 256, 256]            262,400
│    │    │    │    └─Dropout: 5-108               [32, 256, 256]            --
│    └─TransformerBlock: 2-10                      [32, 256, 256]            --
│    │    └─LayerNorm: 3-37                        [32, 256, 256]            512
│    │    └─MultiHeadAttention: 3-38               [32, 256, 256]            --
│    │    │    └─ModuleList: 4-37                  --                        --
│    │    │    │    └─AttentionHead: 5-109         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-110         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-111         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-112         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-113         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-114         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-115         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-116         [32, 256, 32]             24,576
│    │    │    └─Linear: 4-38                      [32, 256, 256]            65,792
│    │    │    └─Dropout: 4-39                     [32, 256, 256]            --
│    │    └─LayerNorm: 3-39                        [32, 256, 256]            512
│    │    └─FeedForward: 3-40                      [32, 256, 256]            --
│    │    │    └─Sequential: 4-40                  [32, 256, 256]            --
│    │    │    │    └─Linear: 5-117                [32, 256, 1024]           263,168
│    │    │    │    └─ReLU: 5-118                  [32, 256, 1024]           --
│    │    │    │    └─Linear: 5-119                [32, 256, 256]            262,400
│    │    │    │    └─Dropout: 5-120               [32, 256, 256]            --
│    └─TransformerBlock: 2-11                      [32, 256, 256]            --
│    │    └─LayerNorm: 3-41                        [32, 256, 256]            512
│    │    └─MultiHeadAttention: 3-42               [32, 256, 256]            --
│    │    │    └─ModuleList: 4-41                  --                        --
│    │    │    │    └─AttentionHead: 5-121         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-122         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-123         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-124         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-125         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-126         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-127         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-128         [32, 256, 32]             24,576
│    │    │    └─Linear: 4-42                      [32, 256, 256]            65,792
│    │    │    └─Dropout: 4-43                     [32, 256, 256]            --
│    │    └─LayerNorm: 3-43                        [32, 256, 256]            512
│    │    └─FeedForward: 3-44                      [32, 256, 256]            --
│    │    │    └─Sequential: 4-44                  [32, 256, 256]            --
│    │    │    │    └─Linear: 5-129                [32, 256, 1024]           263,168
│    │    │    │    └─ReLU: 5-130                  [32, 256, 1024]           --
│    │    │    │    └─Linear: 5-131                [32, 256, 256]            262,400
│    │    │    │    └─Dropout: 5-132               [32, 256, 256]            --
│    └─TransformerBlock: 2-12                      [32, 256, 256]            --
│    │    └─LayerNorm: 3-45                        [32, 256, 256]            512
│    │    └─MultiHeadAttention: 3-46               [32, 256, 256]            --
│    │    │    └─ModuleList: 4-45                  --                        --
│    │    │    │    └─AttentionHead: 5-133         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-134         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-135         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-136         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-137         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-138         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-139         [32, 256, 32]             24,576
│    │    │    │    └─AttentionHead: 5-140         [32, 256, 32]             24,576
│    │    │    └─Linear: 4-46                      [32, 256, 256]            65,792
│    │    │    └─Dropout: 4-47                     [32, 256, 256]            --
│    │    └─LayerNorm: 3-47                        [32, 256, 256]            512
│    │    └─FeedForward: 3-48                      [32, 256, 256]            --
│    │    │    └─Sequential: 4-48                  [32, 256, 256]            --
│    │    │    │    └─Linear: 5-141                [32, 256, 1024]           263,168
│    │    │    │    └─ReLU: 5-142                  [32, 256, 1024]           --
│    │    │    │    └─Linear: 5-143                [32, 256, 256]            262,400
│    │    │    │    └─Dropout: 5-144               [32, 256, 256]            --
│    └─LayerNorm: 2-13                             [32, 256, 256]            512
├─Linear: 1-4                                      [32, 256, 65]             16,705
====================================================================================================
Total params: 9,567,297
Trainable params: 9,567,297
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 320.83
====================================================================================================
Input size (MB): 0.07
Forward/backward pass size (MB): 2252.93
Params size (MB): 38.27
Estimated Total Size (MB): 2291.27
====================================================================================================
The Decoder model has 9,567,297 trainable parameters
```