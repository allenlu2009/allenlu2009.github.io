---

title: AI for AI (II) - Jupyter-ai
date: 2024-02-07 23:10:08
categories:
- AI
tags: [Jupyter]
typora-root-url: ../../allenlu2009.github.io
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>


## 透過 Jupyter-AI 探索 AI 開發

Jupyter notebooks已成為使用 Python 和資料科學 (尤其是開發AI和機器學習模型) 的熱門平台。以下概述了如何使用Jupyter-AI 進行AI編程。[Jupyter-AI](https://github.com/deepjyoti30/jupyter-ai) 是一個令人振奮的工具，擴展了 Jupyter Notebooks 的功能，使開發人員和數據科學家能夠無縫地使用人工智慧（AI）框架。我們將逐步介紹如何設置 Jupyter-AI，提供實際示例，並提供相應的見解。

### 設置 Jupyter-AI

#### 步驟 0：產生 python virtual environment

建議使用 anaconda.

```
> conda create -n ai_jupyter --clone base
> conda activate ai_jupyter
```



#### 步驟 1：安裝 Jupyter-AI (參考 [Jupyter AI documentation](https://jupyter-ai.readthedocs.io/en/latest/users/index.html#model-providers) )

使用 pip 安裝 Jupyter-AI：

```bash
(ai_jupyter) pip install jupyter-ai
```



#### 步驟 2：啟用 Jupyter-AI

安裝完成後，啟用 Jupyter-AI 作為擴展：(**這步可以省略**，因爲 step 1 自動安裝 and enable jupyter_ai)

```bash
(ai_jupyter) jupyter nbextension install --py jupyter_ai
(ai_jupyter) jupyter nbextension enable --py jupyter_ai
(ai_jupyter) jupyter serverextension enable --py jupyter_ai
```



#### 步驟 3：啟動 Jupyter Notebook

如常啟動 Jupyter Notebook：

```bash
(ai_jupyter) jupyter notebook
```

現在，您可以打開 Jupyter Notebook 並體驗 Jupyter-AI 的增強功能。



### 使用 Jupyter-AI：實際示例

1. 最快的方法是 clone jupyter-ai github.  

```
(ai_jupyter) git clone https://github.com/jupyterlab/jupyter-ai.git
```

2. 申請 Open-ai 的 API-KEY

   

#### Ex1: 自動編程 (Code Generation) 使用 ChatGPT 3.5 Turbo

打開 github 的 code.ipynb.

* 首先要設定 OPENAI_API_KEY
* 接著使用咒語 :  ai chatgpt --format code,  描述想要產生的 code.

```python
%reload_ext jupyter_ai
%env OPENAI_API_KEY=xxx

%%ai chatgpt --format code
A function that computes the lowest common multiples of two integers, and a function that runs 5 test cases of the lowest common multiple function
```

```python
def lowest_common_multiple(a, b):
    def gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a

    return (a * b) // gcd(a, b)

def run_test_cases():
    test_cases = [(12, 18), (15, 25), (9, 6), (8, 12), (20, 30)]
    for a, b in test_cases:
        print(f"Lowest common multiple of {a} and {b}: {lowest_common_multiple(a, b)}")

run_test_cases()

def lcm(x, y):
    if x > y:
        greater = x
    else:
        greater = y

    while True:
        if (greater % x == 0) and (greater % y == 0):
            lcm = greater
            break
        greater += 1

    return lcm

def test_lcm():
    assert lcm(3, 5) == 15
    assert lcm(7, 9) == 63
    assert lcm(18, 24) == 72
    assert lcm(10, 15) == 30
    assert lcm(12, 16) == 48

test_lcm()
```

```python
%%ai chatgpt --format code
A function that computes the lowest common multiples of two integers, and a function that runs 5 test cases of the lowest common multiple function
```
```python
def lcm(a, b):
    if a > b:
        greater = a
    else:
        greater = b
    while True:
        if greater % a == 0 and greater % b == 0:
            lcm = greater
            break
        greater += 1
    return lcm


def run_test_cases():
    test_cases = [(2, 3), (4, 6), (10, 15), (8, 12), (5, 7)]
    for a, b in test_cases:
        print(f"LCM of {a} and {b}: {lcm(a, b)}")


run_test_cases()
```
```python
%%ai chatgpt --format code
A function compute the Fabunacci sequence

def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        sequence = [0, 1]
        for i in range(2, n):
            sequence.append(sequence[i-1] + sequence[i-2])
        return sequence

print(fibonacci(10))
```



#### Ex2: 生圖 (Image Generation) 使用 Diffusion Model

```
%reload_ext jupyter_ai
%env HUGGINGFACEHUB_API_TOKEN=token
```

```
%%ai huggingface_hub:stabilityai/stable-diffusion-2-1 --format image
It's an astronaut with a boombox
```

<img src="/media/image-20240207211123659.png" alt="image-20240207211123659" style="zoom:33%;" />

### 評論

#### 精簡的開發流程
Jupyter-AI 通過提供對預訓練模型的輕松訪問並簡化了常見 AI 任務所需的代碼，精簡了開發流程。這使開發人員更能集中精力於特定應用，而不是模型實現的細節。

#### 可擴展性
Jupyter-AI 是可擴展的，這意味著您可以根據需要集成其他 AI 框架和模型。這種靈活性使您能夠在 Jupyter Notebooks 內無縫地使用各種機器學習和深度學習庫。

#### 社區和支援
Jupyter-AI 的社區積極參與項目，確保定期更新和不斷改進。如果您遇到任何問題或有特定的請求，可以與社區互動獲得支援。

### 結論

Jupyter-AI 對於在 Jupyter Notebook 環境中進行 AI 項目的開發人員和數據科學家來說是一個有價值的工具。其易用性、可擴展性和積極的社區支援使其成為您的 AI 開發工具箱中強大的補充。試試看，探索可能性，並用 Jupyter-AI 增強您的 AI 開發工作流程。



## Reference

Jupyterlab. “Jupyterlab/Jupyter-Ai.” Python. 2023. Reprint, JupyterLab, February 7, 2024. https://github.com/jupyterlab/jupyter-ai.

