

## Table of Content

- [[#Vs Code Python and Colab Jupyter Notebook Co-work|Vs Code Python and Colab Jupyter Notebook Co-work]]
	- [[#Vs Code Python and Colab Jupyter Notebook Co-work#How to Avoid Two Copy of Files?|How to Avoid Two Copy of Files?]]
		- [[#How to Avoid Two Copy of Files?#Mount Local Folder|Mount Local Folder]]
		- [[#How to Avoid Two Copy of Files?#Change Directory for Import|Change Directory for Import]]
		- [[#How to Avoid Two Copy of Files?#Import related python and packages|Import related python and packages]]
		- [[#How to Avoid Two Copy of Files?#Command Line Argement|Command Line Argement]]
- [[#Languge Model Visualization|Languge Model Visualization]]



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
use argparse.Namespace as a placeholder and redefine the args using it.

```python
#args = argparse.Namespace(generate="Before we proceed any further")
args = argparse.Namespace(generate=False)
```


## Languge Model Visualization

