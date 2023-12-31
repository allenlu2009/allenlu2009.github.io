---
title: Python Project Management - Structure
date: 2022-10-07 23:10:08
categories:
- Language
tags: [Python, CoPilot]
typora-root-url: ../../allenlu2009.github.io

---

[@liaoPythonImport2020] 指出 import 常常遇到的問題

[@loongProjectStructure2021]  Python project structure 寫的非常好，本文直接引用作爲自己參考。



由於 Python 簡單易用，很多開始使用 Python 的人都是從一個 script 檔案開始，逐步形成多個 Python 檔案組成的程序。

在脫離 Python 幼幼班準備建立稍大型的專案的時候，學習如何組織化 Python 專案是一大要點。分成三個部分：

1. 檔案放在同一個 directory 形成一個 package 打包。對應下面的簡單結構。
2. 不同的 sub-packages 再用一個 (src) directory.  之後一起打包。對應下面的 src 結構的 src directory.
3. Testing 非常重要，但是一般放在分開的 tests directory, 避免被打包。對應下面的 src 結構的 tests directory.



## Module and Package

Python 提供的 **module（模組）**與 **package（套件）**是建立架構的基本元件，但在module之間為了重複使用一些 function（函數）或 class（類別）而必須互相 **import（匯入）**，使用上一個不注意就會掉入混亂的 import 陷阱。

基本上一個檔案就是一個 module，裡頭可以定義 function，class，和 variable。
**把一個 module 想成一個檔案，那一個package就是一個目錄了**。Package 可裝有 subpackage 和 module，讓你的專案更條理更組織化，最後一坨打包好還能分給別人使用。

### 同一個目錄 run 或 import module (檔案)

#### Module (檔案)

先看看 module。假設有一個 module `sample_module.py` 裡頭定義了一個 function `sample_func`：

```python
def sample_func():
    print('Hello!')
```

現在你在**同一個目錄**裡下有另一個 module `sample_module_import.py` 想要使用這個 function，這時可以直接從 `sample_module` import 拿取：

<u>第一種寫法</u> (implicit relative import)

```python
from sample_module import sample_func

if __name__ == '__main__':
    sample_func()
```

跑 `python3 sample_module_import.py` 會得到：Hello!



#### Package

再來是 package。我們把上面兩個檔案包在一個新的目錄 `sample_package` 底下：

```
sample_package/
├── __init__.py
├── sample_module.py
└── sample_module_import.py
```

很重要的是新增那個 `__init__.py` 檔。它是空的沒關係，但一定要有，有點宣稱自己是一個 package 的味道。

這時候如果是進到 `sample_package` 裡面跑一樣的指令，那沒差。但既然都打包成 package 了，通常是在整個專案的其他地方需要用到的時候 import 它，這時候裡面的 import 就要稍微做因應。



### 不同目錄 (up directory) run 或 import module (檔案)

此時我們引入另外兩種寫法：

<u>第二種寫法</u> (explicit relative import)

```python
from .sample_module import sample_func

if __name__ == '__main__':
    sample_func()
```

<u>第三種寫法</u> (absolute import)

```python
from sample_package.sample_module import sample_func

if __name__ == '__main__':
    sample_func()
```



我們修正一下 `sample_package/sample_module_import.py` 。假設這時我們在跟 `sample_package` 同一個 folder 底下執行下面兩種指令：

```
指令 1. $ python3 sample_package/sample_module_import.py  // 需要搭配第一種 implicit relative import 
指令 2. $ python3 -m sample_package.sample_module_import // 需要搭配第二種 explicit relative import
```

以下幾種不同的 import 寫法，會各有什麼效果呢？

```
# 不標準的 implicit relative import 寫法
from sample_module import sample_func
指令 1. 成功印出 Hello!
指令 2. ModuleNotFoundError。

# 標準的 explicit relative import 寫法
from .sample_module import sample_func
指令 1. 包含相對路徑的檔案不能直接執行，只能作為 module 被引用，所以失敗
指令 2. 成功印出 Hello!

# 標準的 absolute import 寫法
from sample_package.sample_module import sample_func
指令 1. 如果此層目錄位置不在 python path (i.e. $PYTHONPATH) 中，就會失敗
指令 2. 成功印出 Hello!
```



執行指令中的 `-m` 代表 module 是為了讓 Python 預先 import 你要的 package 或 module 給你，然後再執行 script。所以這時 `sample_module_import` 在跑的時候，是以 `sample_package` 為環境的，這樣那些 import 才會合理。

另外，[**pythonpath**](https://docs.python.org/3/library/sys.html#sys.path) 是 Python 查找 module 時候使用的路徑，例如 standard module 所在的目錄位置。因此在第三種寫法中，Python 會因為在 python path 中找不到 `sample_package.sample_module`而噴 error。**你可以選擇把當前目錄加到 `sys.path`，也就是 Python path（初始化自環境變數`PYTHONPATH`），來讓 Python 搜尋得到這個 module ，但這個方法很髒很難維護，最多用來debug，其他時候強烈不建議使用。**



**因為常常會用 VS code debug, 因此還是有機會用到 PYTHONPATH, 如何在 VS Code 設定? 有兩種方法：**

1. 直接在 launch.json 設定如下。此處是相當于設定 PYTHONPATH = "./src"  也就是 VS Code {workspaceRoot/src} folder.  
2. 第二種方法是 VS code default 會 load {workspaceRoot}/.env.  也可以用 launch.json 的 envFile 設定 path (這裡也是 ./.env)

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {"PYTHONPATH": "./src"},
            //"env": {"PYTHONPATH":"${workspaceRoot}/src"},  // same as "./src"
            //"envFile": "${workspaceRoot}/.env",
            //"python": "${command:python.interpreterPath}",
            "justMyCode": true
        }
    ]
}
```

.env file content 就只有一行：

```
PYTHONPATH=./src
```



如果要在 command window 執行 python program, 例如 pytest:

在 PC Windows 10 PowerShell (PS), 必須這樣設定 PYTHONPATH:

```
 $env:PYTHONPATH = ".\src"
```

注意：1. 要在 Anaconda 的 PowerShell;  2. 要包含 "$"

在 Mac OS or linux, 可以這樣設定 PYTHONPATH:

```
 $export PYTHONPATH='./src'
```



#### 一個例子: nanoGPTplus

這是一個 github 非常好的例子，使用 Poetry 建構的 pyproject.toml。[GitHub - Andrei-Aksionov/nanoGPTplus](https://github.com/Andrei-Aksionov/nanoGPTplus)



```bash
nanoGPTplus/
├── README.md
├── data
│   └── raw
│       └── tiny_shakespeare
│           └── input.txt
├── logs
│   ├── generation.log
│   └── training.log
├── models
│   ├── gpt_model_small.pth.tar
│   └── tokenizers
│       ├── tokenizer_gpt_large.pkl
│       └── tokenizer_gpt_small.pkl
├── notebooks
│   ├── EDA
│   │   └── tiny_shakespeare.ipynb
│   └── examples
│       ├── bigram_model_training.ipynb
│       ├── gpt_model_training.ipynb
│       └── run_on_google_colab.ipynb
├── pyproject.toml
├── src
│   ├── __init__.py
│   ├── config
│   │   └── config.yaml
│   ├── data
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── downloader.py
│   │   ├── scripts
│   │   │   └── download_tiny_shakespeare.py
│   │   └── tokenizer.py
│   ├── model
│   │   ├── __init__.py
│   │   ├── bigram_language_model
│   │   │   ├── README.md
│   │   │   └── bigram.py
│   │   ├── generate.py
│   │   ├── gpt_language_model
│   │   │   ├── README.md
│   │   │   ├── attention.py
│   │   │   ├── feed_forward.py
│   │   │   ├── gpt.py
│   │   │   ├── peft
│   │   │   │   ├── README.md
│   │   │   │   └── lora.py
│   │   │   └── transformer_block.py
│   │   ├── lr_schedulers.py
│   │   ├── train.py
│   │   └── trainer.py
│   └── utils
│       ├── __init__.py
│       ├── arguments.py
│       ├── config.py
│       ├── device.py
│       ├── error.py
│       ├── model.py
│       └── seed.py
└── tests
    └── smoke
        ├── dataset_test.py
        ├── generate_test.py
        ├── model_test.py
        └── train_test.py
```





**執行 train.py 有 Error.**

```bash
(llama2)> python src/model/train.py gpt --size small
```

**Error 如下。找不到 src path**

```bash
Traceback (most recent call last):
  File "/mnt/c/Users/allen/OneDrive/ml_code/work/nanoGPTplus/src/model/train.py", line 11, in <module>
    from src import config
ModuleNotFoundError: No module named 'src'
```

**若是如下 Training 如下則 OK!**

```bash
(llama2)> python -m src.model.train gpt --size small
```



我們看一下 train.py 的 import package and module 的 path。 

```python
from src import config
from src.data import CharTokenizer, NextTokenDataset
from src.model import (
    BigramLanguageModel,
    CosineWarmupLRScheduler,
    GPTLanguageModel,
    Trainer,
)
from src.model.gpt_language_model.peft.lora import lora, mark_only_lora_as_trainable
from src.utils import (
    RangeChecker,
    get_device,
    get_model_config,
    grab_arguments,
    pickle_dump,
    set_seed,
)

```

* 采用是 absolute path
* src 被加入 absolute path 之中!  
* 所以如果要在 command window 執行 python program, 例如 pytest:

**在 PC Windows 10 PowerShell (PS), 必須這樣設定 PYTHONPATH:**

```
 $env:PYTHONPATH = ".\"   ## 不是 ".\src"
```

注意：1. 要在 Anaconda 的 PowerShell;  2. 要包含 "$"

**在 Mac OS or linux, 要這樣設定 PYTHONPATH:**

```
 $export PYTHONPATH='./'    ## 不是 "./src"
```



**在 VS Code 則是**

1.  在 launch.json 

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {"PYTHONPATH": "./"},
            //"env": {"PYTHONPATH":"${workspaceRoot}"},  // same as "./"
            //"envFile": "${workspaceRoot}/.env",
            //"python": "${command:python.interpreterPath}",
            //"args": ["gpt", "--size", "small", "--max-new-tokens", "500"]
            "args": ["gpt", "--size", "small"],
            "justMyCode": true
        }
    ]
}
```

2. 第二種方法是 VS code default 會 load {workspaceRoot}/.env.  也可以用 launch.json 的 envFile 設定 path (這裡也是 ./.env).  .env 的内容就是一行。

   ```
   PYTHONPATH=./
   ```

   

3.  **在 VS Code debug python -m test 的方法如下：**

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Module",
            //"name": "Python: Current File",
            "type": "python",
            "request": "launch",
            //"program": "${file}",
            //"console": "integratedTerminal",
            "env": {"PYTHONPATH": "./"},
            //"module": "src.model.generate",
            //"args": ["gpt", "--size", "small", "--max-new-tokens", "500"]
            "module": "src.model.train",
            "args": ["gpt", "--size", "small", "--max-new-tokens", "500"]
            //"justMyCode": true
        }
    ]
}
```





#### 基本 import

前面有看過了，這邊統整介紹一下。如果你想使用在其他 module 裡定義的 function、class、variable 等等，就需要在使用它們之前先進行 import。通常都會把需要 import 的 module 們列在整個檔案的最一開始，但不是必須。

**語法1：**`import [module]`

```
# Import 整個 `random` module
import random

# 使用 `random` module 底下的 `randint` function
print(random.randint(0, 5))
```

**語法2：**`from [module] import [name1, name2, ...]`

```
# 從 `random` module 裡 import 其中一個 function `randint`
from random import randint

# 不一樣的是，使用 `randint` 的時候就不需要先寫 `random` 了
print(randint(0, 5))
```

**語法3：**`import [module] as [new_name]`

```
# Import 整個 `random` module，
# 但這個名字可能跟其他地方有衝突，因此改名成 `rd` 
import random as rd

# 使用 `rd` 這個名稱取代原本的 `random`
print(rd.randint(0, 5))
```

**語法4（不推薦）：**`from [module] import *`

```
# Import 所有 `random` module 底下的東西
from random import *

# 使用 `randint` 的時候也不需要先寫 `random`
print(randint(0, 5))
```

**語法4不推薦原因是容易造成名稱衝突，降低可讀性和可維護性。**



#### Absolute Import v.s. Relative Import

Python 有兩種 import 方法，**absolute import** 及 **relative import**。Absolute import 就是完整使用 module 路徑，relative import 則是使用以當前 package為參考的相對路徑。

Relative import 的需求在於，有時候在改變專案架構的時候，裡面的 package 和 module 會拉來拉去，這時候如果這些 package 裡面使用的是relative import 的話，他們的相對關係就不會改變，也就是不需要再一一進入 module 裡更改路徑。但因為 relative import 的路徑取決於當前 package，所以在哪裡執行就會造成不一樣的結果，一不小心又要噴一堆 error；這時absolute import 就會減少許多困擾。

這邊參考[PEP328](https://www.python.org/dev/peps/pep-0328/#guido-s-decision)提供的範例。Package 架構如下 (no src)：

```
package
├── __init__.py
├── subpackage1
│   ├── __init__.py
│   ├── moduleX.py
│   └── moduleY.py
├── subpackage2
│   ├── __init__.py
│   └── moduleZ.py
└── moduleA.py
```

現在假設 `package/subpackage1/moduleX.py`想要從其他 module 裡 import 一些東西，則使用下列語法（`[A]`表 absolute import 範例；`[R]`表 relative import 範例）：

```
# Import 同一個 package 底下的 sibling module `moduleY`
[A] from package.subpackage1 import moduleY
[R] from . import moduleY
[Error] import .moduleY

# 從同一個 package 底下的 sibling module `moduleY` 中，
# import `spam` 這個 function
[A] from package.subpackage1.moduleY import spam
[R] from .moduleY import spam

# 從隔壁 package 底下的 module `moduleZ` 中，
# import `eggs` 這個 function
[A] from package.subpackage2.moduleZ import eggs
[R] from ..subpackage2.moduleZ import eggs

# Import parent package 底下的 module `moduleA`
[A] from package import moduleA
[R] from .. import moduleA 或 from ... package import moduleA
```

要點：

1. Relative import 裡，`..`代表上一層 ，多幾個`.`就代表多上幾層。
2. Relative import 一律採用 `from ... import ...`語法，即使是從 `.` import也要寫 `from . import some_module` 而非 `import .some_module`。原因是`.some_module`這個名稱在 expression 裡無法出現。Absolute import 則無限制。



## Project 檔案結構

大型專案檔案結構 (files and directories) 非常重要。另一個重點是測試 (testing).  因此我希望畢其功於一役，參考專家的結構。

大部分人並沒以一個項目或工程的概念去看待自己的程序。而現在社區中的流行項目也存在兩種不同的目錄結構。

### 1 簡單檔案結構

Python 項目打包 文章中以一個簡單項目結構演示了如何打包一個 Python 項目

```
packaging_tutorial
├── LICENSE
├── README.md
├── example_pkg
│   └── __init__.py
├── setup.py
└── tests
```

項目結構以根目錄開始，作為項目的環境。因為，為了在開發中正常導入 example_pkg 中所有的東西，就需要將項目根目錄添加到 sys.path 中。這也就讓項目根目錄下的所有包都變成了可導入。當有多個同級包時，它們都是扁平的散落在項目根目錄。項目根目錄下可能還存在其他非包目錄，如 data 、 docs 等。如果需要本地引用第三方庫，也需要放到根目錄，但第三方包並不是項目的子包，而是它的一個引用。這樣做會造成混亂。

比如這樣的一個項目：

```
tutorial
├── LICENSE
├── README.md
├── data
|   └── user.json
├── docs
│   └── history.md
├── user
│   └── __init__.py
├── views
│   └── __init__.py
├── requests            # 這是需要本地打包的第三方包
│   └── __init__.py
├── setup.py
└── tests
```
當多個目錄扁平的分佈在項目根目錄時，它們扮演者不同的功能，在開發上，會帶了一定的混亂。而且在打包和測試上也會帶來一些不便。

在打包上，需要提供更多的配置排除不必要的目錄，如 docs 或者其他不需要打包僅項目中的東西。

當使用可編輯安裝（ pip install -e . ） 時，會將項目根目錄中的所有東西安裝到環境中，包括一些不需要的。

使用自動化測試 tox 工具無法檢測安裝之後的問題，因為這種目錄環境可以直接使用環境中的包（項目根目錄被添加到 sys.path 中了）。

### 2 src 結構

Pypa 維護的示例項目 中採用了一種更推薦的結構 src 結構。

```
sampleproject
├── data
├── src
|   └── sample
|       └── __init__.py
├── setup.py
└── tests
```
六年前的這篇文章  [Packaging a python library](https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure)  就詳細闡述了使用 src 結構比簡單結構的諸多有點。而現在也逐漸被社區作為一個標準遵循。雖然社區中有大量老的項目依然採用簡單佈局，但新項目推薦使用 src 結構。

如下面這個示例項目結構：

```
sampleproject
├── data
│   └── user.json
├── docs
│   └── history.md
├── setup.cfg
├── setup.py
├── src
│   ├── requests
│   │   └── __init__.py
│   └── sample
│       ├── __init__.py
│       ├── user
│       │   └── __init__.py
│       └── views
│           └── __init__.py
├── tests
│   ├── __init__.py
│   ├── user
│   │   └── __init__.py
│   └── views
│       └── __init__.py
└── tox.ini
```
項目的包結構很清晰，在環境中只需要引入 src 目錄，就可以輕鬆導入項目原始碼。通過 pip install -e . 可編輯安裝，也只會安裝 src 中的包。管理起來更加清晰。

### 3 實踐
下面以一個簡單真實的項目來演示使用 src 組織項目

#### 3.1 創建項目:


```
mkdir sampleproject
cd sampleproject
```
初始化版本管理：


git init
\# 如果沒有全局用戶名和郵箱，需要先配置
git config user.email example@example.com
git config user.name example
創建項目自述檔案：

```touch README.md```

#### 3.2 編寫項目原始碼

創建項目包：

```
mkdir src/sample_project
touch src/sample_project/__init__.py
```
初始化版本號：

```src/sample_project/__init__.py```


```__version__ = '0.1.0'```

安裝依賴：

```pip install click```

創建命令入口檔案：

```src/sample_project/cmdline.py```

```
import click

@click.command()
def main():
    click.echo('Hello world!')


if __name__ == "__main__":
    main()
```

#### 3.3 編寫測試
創建測試目錄：

```
mkdir -p tests/sample_project
touch tests/sample_project/__init__.py
```

安裝依賴：

```pip install pytest```
創建測試檔案：

```tests/sample_project/test_cmdline.py```

```
from click.testing import CliRunner

from sample_project import cmdline


def test_main():
    runner = CliRunner()
    result = runner.invoke(cmdline.main)
    assert 'Hello world!' in result.output
```
設定 PYTHONPATH (使用 absolute path)

```
export PYTHONPATH='/Users/allenlu/OneDrive/ml_code/sampleproject/src'
```

運行測試：

```
#pip install -e .  # 以可編輯安裝方式到環境中
pytest
```
測試運行成功，說明功能正確



#### 3.4 初始化打包配置

編寫打包配置：

```setup.py```

```
import setuptools

setuptools.setup()
```
```
setup.cfg


[metadata]
name = sample_project
version = attr: sample_project.__version__
author = example
author_email = example@example.com
description = Sample Project
keywords = ssl_manager
long_description = file: README.md
long_description_content_type = text/markdown
classifiers =
    Operating System :: OS Independent
    Programming Language :: Python :: 3.7

[options]
python_requires > = 3.7
include_package_data = True
packages = find:
package_dir =
    = src
install_requires =
    click

[options.entry_points]
console_scripts =
    ssl_manager = sample_project.cmdline:main

[options.packages.find]
where = src

[tool:pytest]
testpaths = tests
python_files = tests.py test_*.py *_tests.py
```
打包：

```python setup.py bdist_wheel```



打包之後可以用

> $ pip install -e . 

就可以 pip install 在目前的 python package.  使用 -e 是 in development phase, 會隨 local change 改變。

 [setuptools - Python setup.py develop vs install - Stack Overflow](https://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install)

[什么时候-e，--editable选项对pip安装有用？ (qastack.cn)](https://qastack.cn/programming/35064426/when-would-the-e-editable-option-be-useful-with-pip-install)

使用  conda list 應該就會看到 word_count 這個 package.  版號是放在 \_\_init.py\_\_ 的版號嗎？ YES!

```python
"""Word Count"""
__version__ = '0.1.0'
```



#### 3.5 總結
至此，一個項目開發完成，完整項目結構如下：
```
├── build
│   ├── bdist.linux-x86_64
│   └── lib
│       └── sample_project
│           ├── cmdline.py
│           └── __init__.py
├── dist
│   └── sample_project-0.1.0.linux-x86_64.tar.gz
├── setup.cfg
├── setup.py
├── src
│   ├── sample_project
│   │   ├── cmdline.py
│   │   ├── __init__.py
│   └── sample_project.egg-info
│       ├── dependency_links.txt
│       ├── entry_points.txt
│       ├── PKG-INFO
│       ├── requires.txt
│       ├── SOURCES.txt
│       └── top_level.txt
└── tests
    ├── __init__.py
    └── sample_project
        ├── __init__.py
        └── test_cmdline.py
```



## Reference