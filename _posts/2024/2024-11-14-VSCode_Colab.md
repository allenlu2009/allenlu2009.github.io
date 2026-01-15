---
title: VS Code on Colab
date: 2024-11-14 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-09-15-Useful_Tools]]"
categories:
  - GenAI
tags:
  - Attention
  - IDE
  - Kernel
  - Tools
  - Transformer
---


本文目的很簡單。就是 local VS Code 寫 Python or Jupyter, 但是 remote on Colab 利用其算力。
以下是 ChatGPT 的 answers.

Not working so far!!
Google Colab disallow remote SSH or proxy to access colab, at least for free or low tier users.

## Method B (Not working)

Google Colab recently disabled direct SSH access due to security and policy changes. However, you can still use VS Code as a front-end for Colab by using Colab’s **code-server** setup, which essentially runs a version of VS Code inside the Colab environment and allows access through a browser. Here’s how to set it up:

### Steps to Use VS Code as a Front-End with Google Colab

1. **Start a Google Colab Notebook**:
   - Open a new Colab notebook and make sure your runtime is set to GPU (if you need GPU access).

2. **Install and Launch code-server in Colab**:
   - In a Colab code cell, run the following commands to install **code-server**, a web-based version of VS Code:

     ```python
     !curl -fsSL https://code-server.dev/install.sh | sh
     !mkdir -p ~/.config/code-server/
     ```

   - Next, set a password for accessing the code server:

     ```python
     with open("~/.config/code-server/config.yaml", "w") as f:
         f.write("bind-addr: 0.0.0.0:8080\nauth: password\npassword: 'your_password'\n")
     ```

     Replace `'your_password'` with a secure password you will use to access the code server.

3. **Start code-server**:
   - Start the code-server with the following command:

     ```python
     !code-server --config ~/.config/code-server/config.yaml &
     ```

4. **Create a Public URL to Access code-server**:
   - Colab allows you to use `ngrok` or `localtunnel` to create a public URL for accessing code-server. Use `localtunnel` as it’s simpler to set up:

     ```python
     !npm install -g localtunnel
     !lt --port 8080
     ```

   - The output will display a URL like `https://<unique-subdomain>.loca.lt`. This is your access point to the code-server.

5. **Access code-server in Your Browser**:
   - Open the URL from the `localtunnel` output in your browser. You should see the VS Code interface.
   - Enter the password you set in step 2 to log in.

6. **Use VS Code in Your Browser**:
   - You can now use VS Code directly in your browser, running on Colab’s backend. This setup will give you access to Colab's CPU/GPU/TPU resources while working with a familiar VS Code interface.

7. **Saving Your Work**:
   - Remember to use Google Drive or other cloud storage to save your work, as Colab instances are ephemeral.

This method provides a practical workaround to use VS Code with Google Colab’s resources without needing SSH.

## Method A

To use Visual Studio Code (VS Code) as the front end while running Python code on Google Colab (using Colab's resources), you can set up a remote connection to Colab from VS Code by forwarding Colab’s runtime. Here’s a step-by-step guide:

### Prerequisites
1. Make sure you have Python installed locally.
2. Install the **Remote - SSH** extension in VS Code.

### Steps

1. **Set Up SSH on Colab**:  (NOT working!  Google Colab disallow SSH connection!)
   - Open your Colab notebook and run the following code to set up an SSH connection. Replace `your_password` with a strong password:

     ```python
     # Install colab_ssh
     !pip install colab_ssh --upgrade

     # Import colab_ssh
     from colab_ssh import launch_ssh, init_git

     # Enable SSH
     launch_ssh("your_email@example.com", "your_password")
     ```

   - You’ll see output with connection details. For example:

     ```
     SSH: ssh -L 8888:localhost:8888 root@0.tcp.ngrok.io -pXXXXX
     Password: your_password
     ```

2. **Forward Colab's SSH Port Locally**:
   - On your local machine, open a terminal and enter the SSH command given by Colab. For example:

     ```bash
     ssh -L 8888:localhost:8888 root@0.tcp.ngrok.io -pXXXXX
     ```

     Enter your password when prompted.

3. **Configure VS Code**:
   - In VS Code, open the **Command Palette** (Ctrl+Shift+P) and select **Remote-SSH: Connect to Host**.
   - Enter the SSH command (`root@0.tcp.ngrok.io -pXXXXX`) that Colab provided, and connect.
   - After connecting, VS Code will treat Colab as a remote server.

4. **Set Up Your Workspace**:
   - You can open a folder and start writing code in VS Code as if you were working locally. VS Code will send your code to Colab for execution, leveraging Colab's resources (like GPU or TPU).

5. **Run Jupyter Notebooks** (Optional):
   - If you’re working with Jupyter notebooks, you can access them from your browser by going to `localhost:8888` in your local machine, where the Jupyter server is forwarded.

6. **Using Python Scripts**:
   - Create Python scripts or notebooks as you would normally in VS Code. When you execute them, they will run on Colab’s backend.

### Important Tips
- **Be mindful of idle timeouts on Colab**. If you leave the connection idle for too long, Colab might terminate the session.
- **File persistence**: Files saved in the Colab environment will disappear when the session ends. Use Google Drive or other cloud storage if you need persistence.

This setup allows you to enjoy the rich interface of VS Code while leveraging Google Colab's powerful backend.






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