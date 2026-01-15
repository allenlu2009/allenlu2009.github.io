---
title: Python Project Management - Minbpe
date: 2024-09-27 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2022-10-07-Python_Prj]]"
categories:
  - Tools
tags:
  - Coding
  - Python
  - VScode
---


[@liaoPythonImport2020] 指出 import 常常遇到的問題

[@loongProjectStructure2021]  Python project structure 寫的非常好，本文直接引用作爲自己參考。

[GitHub - karpathy/minbpe: Minimal, clean code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization.](https://github.com/karpathy/minbpe)


## Minbpe Minimal Byte Pair Encoding (BPE)

以下我們用 Karpathy 的 minbpe 爲例。
先從 github 下載 minbpe.  
用 tree 看結構。
用 pytest 確認是否 ok.

```
$ git clone https://github.com/karpathy/minbpe.git
$ tree
.
├── minbpe
│   ├── __init__.py
│   ├── base.py
│   ├── basic.py
│   ├── gpt4.py
│   └── regex.py
├── minbpe.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   └── top_level.txt
├── pyproject.toml
├── requirements.txt
├── setup.py
├── tests
│   ├── __init__.py
│   ├── taylorswift.txt
│   └── test_tokenizer.py
└── train.py

$ pytest
collected 21 items
tests/test_tokenizer.py ........     [100%]
============= 21 passed in 22.04s ===
```


### Method 1:  pip install -e .

如果沒有 pyproject.toml 或是 setupy.py, 必須先產生一個。
此處以 setup.py 爲例。

```python
from setuptools import setup, find_packages

setup(
    name='minbpe',
    version='0.1',
    packages=['minbpe'],  # Explicitly specify the package
    install_requires=[
        # List any dependencies here
    ],
)
```

接下來就執行下式，產生 minbpe 0.1 version.
```
$ pip install -e .
```



### 使用方法

```python
from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

def test_encode_decode_identity(tokenizer, text):
    text = unpack(text)
    #tokenizer = tokenizer_factory()
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert text == decoded

def unpack(text):
    # we do this because `pytest -v .` prints the arguments to console, and we don't
    # want to print the entire contents of the file, it creates a mess. So here we go.
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__))
        taylorswift_file = os.path.join(dirname, text[5:])
        contents = open(taylorswift_file, "r", encoding="utf-8").read()
        return contents
    else:
        return text

tokenizer = GPT4Tokenizer()

test_strings = [
    "", # empty string
    "?", # single character
    "hello world!!!? (안녕하세요!) lol123 😉", # fun small string
    "FILE:taylorswift.txt", # FILE: is handled as a special string in unpack()
]

for text in test_strings:
    test_encode_decode_identity(tokenizer, text)



```


## Reference