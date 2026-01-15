---
title: Coding of Perplexity of LLM
date: 2025-01-15 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-1231-Perplexity]]"
categories:
  - GenAI
tags:
  - Algorithm
  - GPT
  - LLM
  - Metric
  - NLP
  - Perplexity
  - Transformer
---




## Source
- [(22) The Key Equation Behind Probability - YouTube](https://www.youtube.com/watch?v=KHVR587oW8I&t=810s)  a good youtube video of cross-entropy

- https://www.reddit.com/r/MachineLearning/comments/oye64h/r_struggling_to_reproduce_perplexity_benchmarks/
- https://thegradient.pub/understanding-evaluation-metrics-for-language-models/
- https://huggingface.co/transformers/v4.2.2/perplexity.html
- https://stackoverflow.com/questions/79134740/perplexity-very-high-on-wikitext-for-gpt2xl
- https://discuss.pytorch.org/t/perplexity-very-high-on-wikitext-for-gpt2xl/212273

Two things
1. check other pdf cascading!
2. check liger to increase model size
3. do scatter plot
4. check fine-tune result using Shakespeare
5. (done) check Shakespeare dataset batch 32-38 text -> 檢查是 Romeo and Juliet 的故事。應該是太有名被 training 到 3B model 中。
6. print the peak memory and throughput information


## Major Issues:

1. For pre-trained model (e.g. GPT2, Llama-1B/3B), 似乎只有用 maximum length (tokenize the entire text and chunk) to get the correction perplexity.
2. 但是 fine-tuned model, 似乎反而是用 variable length (pad to max?) 才會得到 improvement.  如果是 maximum length, 反而沒有任何 improvement, why?




| **Model**            | ---                                | Pre-  | Train  | ---  | ---              | Fine- | Tune | ---   |
| -------------------- | ---------------------------------- | ----- | ------ | ---- | ---------------- | ----- | ---- | ----- |
|                      | Length                             | Batch | Loss   | PPL  | Length           | Batch | Loss | PPL   |
| GPT-2 (124M)         | 1024                               | 4/70  | 3.36   | 28.7 | 1024             | 8/31  | 3.34 | 28.3  |
|                      |                                    |       |        |      | variable         |       | 0.23 | *1.3* |
|                      |                                    |       |        |      | pad to 1024      | 8/470 | 0.96 | 2.6   |
|                      |                                    |       |        |      | pad to batch max | 8/470 | 1.82 | 6.3   |
| -- use hugging       |                                    |       | 3.23   | 25.2 |                  |       |      |       |
| GPT2-large (774M)    | 1024                               |       | 2.8    | 16.4 |                  |       |      |       |
| LLaMA-1B             | 1024/st=0                          | 4/71  | 2.69   | 14.7 |                  |       |      |       |
|                      | st_ratio=0.5                       | 1/?   | 2.57   | 12.5 |                  |       |      |       |
| --                   | 2048                               | 2/71  | 2.56   | 13.0 |                  |       |      |       |
|                      | st_ratio=0.5                       | 1/?   | 2.45   | 11.6 |                  |       |      |       |
| --                   | 4096                               | 1/71  | 2.5    | 12.1 |                  |       |      |       |
|                      | st_ratio=0.5                       | 1/?   | 2.41   | 11.2 |                  |       |      |       |
| LLaMA-3B             | 1024                               | 4/71  | 2.1    | 8.5  |                  |       |      |       |
| --                   | 2048                               | 2/71  | 2.0    | 7.6  |                  |       |      |       |
| --                   | 4096                               | 1/71  | 1.96   | 7.1  |                  |       |      |       |
| Phi3-mini-3.8B       | 1024                               | 4/82  | 1.93   | 6.9  |                  |       |      |       |
| --                   | 2048                               | 2/82  | 1.82   | 6.2  |                  |       |      |       |
| --                   | 4096                               | 1/82  | 1.75   | 5.8  |                  |       |      |       |
| Gemma-7B             | 1024                               | 4/72  | 5.88?? |      |                  |       |      |       |
| Gemma7B              | 2048                               | 4/36  | --     |      |                  |       |      |       |
| Gemma7B              | 2048                               | 2/72  | 4.6    |      |                  |       |      |       |
|                      |                                    |       |        |      |                  |       |      |       |
| 2025/7/9 use hugging |                                    |       |        |      |                  |       |      |       |
| GPT2-large (774M)    | 1024<br>/stride=512                |       | 2.8    | 16.4 | gpt2perp.ipynb   |       |      |       |
|                      | 1024<br>/stride=1024<br>no overlap |       |        | 19.4 |                  |       |      |       |
| GPT2 (124M)          | 1024<br>/stride=512                |       | 3.22   | 25.2 | gpt2perp.py      |       |      |       |
| GPT2 (124M)          | 1024<br>/stride=1024<br>no overlap |       | 3.4    | 29.9 | gpt2perp.py      |       |      |       |
|                      |                                    |       |        |      |                  |       |      |       |


## Two Common Methods for Perplexity

PP 的公式如下：

$$
PP = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log p(w_i | w_1, w_2, \ldots, w_{i-1})\right) = \exp\left(\text{cross entropy}\right)
$$

### Code 1 (no use): Pytorch on Home-made Neural Network!

除了自己寫 C 或是 Python code, 最直接的方法是用 Pytorch 如下。如果只是一個句子，長度小於 max_length of tokenizer (對於 gpt2 是 1024), 方法如下。先算 logits and labels
- shifted_logits 需要把 model output 最後一個拿掉 logits[:,:-1, :] 和 labels 對齊
- shifted_labels 需要把 inputs 的第一個拿掉。
- P = softmax(shifted_logits) 是每一個 token 所有可能 vocab 的機率分佈，不是上式的 p (已經是一個機率)。
- Cross-entropy loss 才是上式的 -log p (given labels),  如果選錯 label 就會讓對應的 -log p 變大，變成大的 loss (loss 是正數)。  
- 注意這裡的 cross-entropy loss 已經是平均值 (reduction = mean), 對應 perplexity 公式 exp 內的公式。
- 最後再做 exponential
- 下節有 step-by-step 說明.
- 最複雜不是一個句子，而是整個 dataset 如何計算 perplexity?  下節討論。

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load your dataset (e.g., WikiText)
text = "Your sample text from WikiText goes here."
inputs = tokenizer(text, return_tensors="pt")

# Calculate logits
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Compute perplexity
shifted_logits = logits[:, :-1, :].contiguous()
shifted_labels = inputs['input_ids'][:, 1:].contiguous()
log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
loss = loss_fct(log_probs.view(-1, log_probs.size(-1)), shifted_labels.view(-1))
perplexity = torch.exp(loss)

print(f"Perplexity: {perplexity.item()}")

```

### Code 2 (no use): HuggingFace 的 Transformer Network

**如果使用 Hugging Face 支持的 model, 非常貼心，已經直接算出 outputs.loss 如下。**
**我驗證過和上面的多步結果完全一致！所以可以直接使用。**
不過上面 Pytorch 的方法對於 home made neural network 有用！因爲 home made network 沒有 outputs.loss, pytorch 只有輸出 logits.

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Load WikiText-2
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = " ".join(dataset["text"])  # length : 1289979

# Tokenize and prepare inputs
inputs = tokenizer(text, return_tensors="pt", truncation=True)  # truncate to 1024

# Compute loss
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss  # Cross entropy loss
    perplexity = torch.exp(loss)   # only 1024 token perplexity

print(f"Perplexity: {perplexity.item()}")

```


### (New 2025/7/12):  Code 3: Huggingface new code for computing Perplexity

Code 2 是指如何用 HuggingFace outputs.loss function 可以直接得到 cross entropy.  但只 truncate 到 1024 token.  用 input and shifted input 計算 1023 tokens loss function.  顯然不是我們期待利用全部 token 的結果。

實務上要解決兩個問題 (1) 利用所有的 tokens，但是不可能一次處理非常長 tokens, 所以要切 chunks; (2) perplexity 和 token 在 token chunk 的位置有關。一開始位置的 token perplexity 會比較差，因爲可能性很多。但是愈後面 接近 max length token position 的 perplexity 會變小，最後會 saturate 到這個 LLM 預測下一個 token 的真實能力。我們其實要比較的就是這個能力。  

Huggingface 提供一個 sample code 如何利用全部 tokens for cross entropy 計算。


```python
nll_sum = 0.0
n_tokens = 0
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss
    # Accumulate the total negative log-likelihood and the total number of tokens
    num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
    batch_size = target_ids.size(0)
    num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
    nll_sum += neg_log_likelihood * num_loss_tokens
    n_tokens += num_loss_tokens
    prev_end_loc = end_loc
    if end_loc == seq_len:
        break
avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
ppl = torch.exp(avg_nll)\n explain the code
```

## Overview

The code processes a sequence that's longer than the model's maximum context length (1024 in the case of GPT2) by breaking it into **overlapping chunks** and computing **the average negative log-likelihood (NLL)** across all tokens, then converting that to perplexity.

## Key Variables

- `stride`: Step size for the sliding window (typically smaller than `max_length` to create overlap)
- `max_length`: Maximum sequence length the model can handle
	- GPT2 max_length = 1024,  stride 512 代表一半 overlap
	- stride=1024 代表 no overlap
- `seq_len`: Total length of the input sequence
	- wikitext2 tokens 的全部長度
- `nll_sum`: Accumulates total negative log-likelihood
- `n_tokens`: Counts total tokens processed

## Step-by-Step Process

**1. Sliding Window Setup**

```python
for begin_loc in range(0, seq_len, stride):
    end_loc = min(begin_loc + max_length, seq_len)
```

Creates overlapping windows of size `max_length`, moving by `stride` each time.
檢查最後一個 window 如果小於 1024 (+max_length).

**2. Target Preparation (避免 stride windows token 被計算兩次)**

```python
trg_len = end_loc - prev_end_loc
target_ids = input_ids.clone()
target_ids[:, :-trg_len] = -100
```

- Only the "new" tokens (from `prev_end_loc` to `end_loc`) are used for loss calculation
- Earlier tokens are masked with `-100` so they don't contribute to the loss
- **This prevents double-counting tokens in overlapping regions**

**3. Loss Calculation**

```python
outputs = model(input_ids, labels=target_ids)
neg_log_likelihood = outputs.loss
```

**The model computes cross-entropy loss only on the unmasked tokens.**

**4. Token Counting Adjustment**

```python
num_valid_tokens = (target_ids != -100).sum().item()
num_loss_tokens = num_valid_tokens - batch_size
```

Subtracts `batch_size` because the model internally shifts labels left by 1 position (each token predicts the next token), so there's one fewer prediction per sequence.  一般 batch_size =1??

**5. Accumulation**

```python
nll_sum += neg_log_likelihood * num_loss_tokens
n_tokens += num_loss_tokens
```

Accumulates the total NLL and token count across all windows.

**6. Final Calculation**

```python
avg_nll = nll_sum / n_tokens
ppl = torch.exp(avg_nll)
```

Computes average NLL per token, then converts to perplexity using the exponential function.

The value `-100` is used as a mask because it has **special meaning in PyTorch's CrossEntropyLoss function**.

## How CrossEntropyLoss Handles -100

When PyTorch's `CrossEntropyLoss` encounters a target label of `-100`, it:

1. **Ignores that token completely** - doesn't include it in loss calculation
2. **Doesn't count it** toward the averaging denominator
3. **Treats it as "no prediction needed"**

This is hardcoded behavior in PyTorch - `-100` is the default `ignore_index` parameter.

## Why This Matters for Sliding Windows

In the sliding window approach:

```python
# Example with stride=2, max_length=4
# Window 1: tokens [0,1,2,3] - calculate loss on all 4 tokens
# Window 2: tokens [2,3,4,5] - calculate loss on only tokens [4,5]
```

For Window 2, you want to:

- **Use tokens [2,3]** as context (they help predict tokens 4,5)
- **Not calculate loss** on tokens [2,3] (already counted in Window 1)
- **Only calculate loss** on the new tokens [4,5]

So the code does:

```python
target_ids = [2, 3, 4, 5]           # Original
target_ids = [-100, -100, 4, 5]    # After masking
```




## Steps to Compute Perplexity using Pytorch

### 1. **加載模型和 tokenizer**
加載預訓練的語言模型及其標記器：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"  # 替換為您的模型名稱
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

### 2. **對輸入文本進行 tokenization**
準備要計算困惑度的文本：
```python
text = "您的輸入文本在這裡。"
inputs = tokenizer(text, return_tensors="pt")  # 使用 'pt' 代表 PyTorch 或 'tf' 代表 TensorFlow
```

### 3. **計算邏輯和對數概率**
運行模型以獲得邏輯並計算對數概率：
```python
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 移動邏輯和標籤以對齊預測與實際標記
shifted_logits = logits[:, :-1, :].contiguous()
shifted_labels = inputs['input_ids'][:, 1:].contiguous()

# 計算對數概率
log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
```

### 4. **計算困惑度**
計算平均負對數似然，然後取指數以獲得困惑度：
```python
# 收集實際標記的對數概率
loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
loss = loss_fct(log_probs.view(-1, log_probs.size(-1)), shifted_labels.view(-1))

# 計算困惑度
perplexity = torch.exp(loss)
print(f"困惑度: {perplexity.item()}")
```

### 6. **處理長輸入或是多 records Dataset (參考 Code 3!)**
對於超過模型上下文長度的長文本：
- 將文本拆分為適合模型最大輸入大小的小塊。
- 計算每個小塊的困惑度，並將其平均為序列級別得分。


## Key：整個 dataset 如何計算 perplexity?

我們以 Model GPT2 (124M parameter), Dataset Wikitext2 為例，其 split:  train / val / test = 36718 / 3760 / 4358 records

### Table of Evaluation Methods and Results for Wikitext

以下所有的 max context length = 1024, 因為這是 GPT2 最大的 position encoding length.
GPT2 vocab size 是 50257.  **嚴格來說，perplexity maximum 是 vocab size.**

| **Method**                                                  | **Model** | **Avg Loss** | **Perplexity** | avg loss/PPL after fine tune |
| ----------------------------------------------------------- | --------- | ------------ | -------------- | ---------------------------- |
| 1. (Code1/2) Long context, truncate to 1024 tokens          | GPT-2     | 2.8          | 15.9           |                              |
| 2. Compute loss per record (uneven length), average evenly  | GPT-2     | 4.8          | 121            |                              |
| 3. Compute loss per record (padded to 1024), average evenly | GPT-2     | 10.17        | 26148          | 1.22/3.4 or 0.96/2.6         |
| 4. Filter out records < 100 tokens, average loss evenly     | GPT-2     | 3.7          | 40             |                              |
| 5. Average loss by token (token-weighted)                   | GPT-2     | 3.9          | 50             | 72-why worse?                |
| 6. Combined records into 1024-token samples with stride=512 | GPT-2     | 3.35         | 28.6           | 30?                          |
| 7. Same as 6 but change stride=0 to save computation        | GPT-2     | 3.36         | 28.7           | 30?                          |
| 8.  (Code3) 用 HuggingFace 例子, stride=1024, no overlap       | GPT-2     | 3.4          | 29.9           | 28.66 if using " "連接         |
| **9.  (Code3) 用 HuggingFace 例子, stride=512**, best case!    | GPT-2     | 3.22         | 25.2           | 24 if using " "連接            |
|                                                             |           |              |                |                              |

1.  (不合理) Truncate entire dataset to 1024 at tokenizer:
- 使用 wikitext2 的 test split: 只有 4358 records.
- 接下來 cascade 4358 records "text",  長度是 1289979 characters
- 在 tokenizer 自動 truncate 成 [1, 1024] tokens!
- outputs.logits.shape = [1, 1024, 50257]. 50257 是 vocab size.
- outputs.loss = 2.7677
- Final perplexity = exp(2.7677) = 15.9!    
- 比較 GPT2 fine-tuning 的 loss 大約是 0.34 (training, maybe overfit?), perplexity = exp(0.34) ~ 1.4). --> fine-tune 之後的 perplexity 的確會變小非常多？
- 顯然只做一個 block (1024) 沒有很好的平均效果。

2. (bad, 有平均但短句 dominate) compute each records (skip 0, range 1-500 words with or without truncation to 1024 tokens) loss separately with variable token length and average each record loss evenly. avg loss: 4.8, Perplexity: 121

3. (terrible!, pad to 1024 非常差!) compute each records (pad to 1024 tokens) loss separately with variable token length, avg loss: 12, Perplexity: 1111111!


我發現短的句子的 loss 比較大，這很合理，因為一開始猜什麼字很發散 perplexity 比較大。但是字數越多，就約容易猜下一個字。以下是 scatter plot of gpt2 on Wikitext.  
4. (better 不過實務上不可能設一個 threshold):  filter out 100 words 以下的字，再 average 剩下 records loss evenly.  avg loss: 3.7,  perplexity: 40
5. (以 token 為單位, 加權平均 loss, 短句的權重小，長句的權重大):  前面都是 average record evenly, 其實不合理。應該是以 token 為單位做 average.  長的句子的 token 比較多，佔的權重大。就是回覆成原始的公式計算 token by token.  可以得到 avg loss:3.9,  perplexity: 50.
6. (correct way!):  應該是把所有的 records 或是句子串在一起，然後切除 1024 token 為一個單位，每次 overlap, i.e. stride=512 token 確保 context 的連貫性。但實務上因為我們只有 words, 需要先全部轉換成 tokens, 再切成 1024 tokens,  而且可以 pack 成 batch 以加速。可以得到 avg loss: 3.35, perplexity: 28.6 @ stride=512, or avg loss: 3.36, perplexity: 28.7 @ stride=0



GPT2: 124M parameter (stride = 512). avg loss: 3.4, perplexity: 28.7
![[Pasted image 20250104002831.png]]

### Perplexity of Wikitext2  for Different Models

接下來我們用 method 7: 串成一個 long text, tokenization, 切除 max context length (1K/2K/4K), pack 成 tensors in batch mode.   計算 perplexity vs. model and context length.  結果如下。

- Model size 越大，avg loss 越小，perplexity 越小。這是 scaling law. 
	- GPT2 < Llama-1B < Llama-3B < Phi3-3.8B < ???
	- 但是在 Gemma-7B 看起來有問題，需要看是否是舊的 model.
- Stride = 0 or 512 沒有影響。所以都用 stride = 0 可以節省計算.
- Batch size = 4, 2, 1 沒有影響。原則越大越好節省計算。但會有 OOM (Out Of Memory) issue.  
	- 主要是最後的 lm_head 問題 (use Liger to solve this bottleneck?)
	- 以 batch = 4, context length = 4K, vocab size = 50K, BF16 為例:   4 x 4000 x 50000 x 2 =  1.6 GB.  Gemma vocab = 256K, 如果 batch=8,  就會有 16GB!  再加上 model 14B, 所有 dynamic memory, 很容易就爆掉！
- Longer context length, lower perplexity.  不過 diminish return,  基本 4K context length 應該就 OK.


以下是 Wiki2 Dataset for different models and max_length/batch_size

| **Model**      | Model/All GB | max_length | Batch size/ Num | **Avg Loss** | **Perplexity** |
| -------------- | ------------ | ---------- | --------------- | ------------ | -------------- |
| GPT-2 (120M)   |              | 1024/s0    | 4/70            | 3.36         | 28.7           |
| --             |              | 1024/s512  | 4/70            | 3.35         | 28.6           |
| LLaMA-1B       |              | 1024       | 4/71            | 2.69         | 14.7           |
| --             |              | 2048       | 2/71            | 2.56         | 13.0           |
| --             |              | 4096       | 1/71            | 2.5          | 12.1           |
| LLaMA-3B       |              | 1024       | 4/71            | 2.1          | 8.5            |
| --             |              | 2048       | 2/71            | 2.0          | 7.6            |
| --             |              | 4096       | 1/71            | 1.96         | 7.1            |
| Phi3-mini-3.8B | 7.7/21GB     | 1024       | 4/82            | 1.93         | 6.9            |
| --             |              | 2048       | 2/82            | 1.82         | 6.2            |
| --             |              | 4096       | 1/82            | 1.75         | 5.8            |
| Gemma-7B       | 37-43GB      | 1024       | 4/72            | 5.88??       |                |
| Gemma7B        | OOM          | 2048       | 4/36            | --           |                |
| Gemma7B        | 42GB         | 2048       | 2/72            | 4.6          |                |

##### **Phi3-mini-4K (3.8B)**  vs.  Llama-3B on Wiki2 dataset 比較
- Phi3-3.8B
	- 紅色: block_size: 1024, avg loss: 1.93, perplexity: 6.9
	- 紫色: block_size: 2048, avg loss: 1.82, perplexity: 6.2
	- 褐色: block_size: 4096, avg loss: 1.75, perplexity: 5.8
- **Llama-3B** 
	- 藍色: block_size: 1024, avg loss: 2.1, perplexity: 8.5
	- 橙色: block_size: 2048, avg loss: 2.0, perplexity: 7.6
	- 綠色: block_size: 4096, avg loss: 1.96, perplexity: 7.1
![[Pasted image 20250107005124.png]]



### Perplexity of Different Datasets (Wiki2, PTB, Shakespeare) vs. Different Models

- Dataset 難度:  Wiki2 < PTB < Shakespeare
- 小模型 (GPT2 and Llama-1B) 所有 batch 都一樣差
- 大一點模型 (Llama-3B and Phi3-3.8B) 在 Shakespeare dataset batch 30-38 有特別好，why?
	- 後來我檢查是 30: Romeo and Juliet 的故事。應該是被大一點的 model training dataset contamination.

| **Model**         | max_length | Wiki2     | **PTB**   | **Shakepseare** | Wiki2 fine-tune      |
| ----------------- | ---------- | --------- | --------- | --------------- | -------------------- |
| GPT-2 (120M)      | 1024       | 3.36/28.7 | 3.87/47.8 | 4.17/64.6       | 0.89(512)/0.47(1024) |
| GPT2-large (1.6B) | 1024       | 2.88/17.8 | 3.42/30.6 | 3.67/39.4       |                      |
| LLaMA-1B          | 4096       | 2.5/12.0  | 3.18/24.1 | 3.44/31.5       |                      |
| LLaMA-3B          | 4096       | 1.96/7.1  | 2.54/12.8 | 2.21/9.13       |                      |
| Phi3-3.8B         | 4096       | 1.75/5.8  | 2.5/12.2  | 2.49/12.0       |                      |
|                   |            |           |           |                 |                      |

GPT2 (120M) on Wiki2, PTB, and Shakespeare datasets!
![[Pasted image 20250108183921.png]]

GPT2-large (1.6B) on Wiki2, PTB, and Shakespeare datasets!
![[Pasted image 20250108210929.png]]


**Llama-1B on wikitext2** (non-uniform dataset; some are easier)
- 藍色: block_size: 1024, avg loss: 2.7, perplexity: 14.7
- 橙色: block_size: 2048, avg loss: 2.6, perplexity: 13.0
- 綠色: block_size: 4096, avg loss: 2.5, perplexity: 12.0
**Llama-1B on PTB** (uniform dataset)
- 藍色: block_size: 1024, avg loss: 3.34, perplexity: 28.3
- 橙色: block_size: 2048, avg loss: 3.25, perplexity: 25.7
- 綠色: block_size: 4096, avg loss: 3.18, perplexity: 24.1
**Llama-1B on Shakespeare** (uniformly bad dataset)
- 藍色: block_size: 1024, avg loss: 3.56, perplexity: 35.3
- 橙色: block_size: 2048, avg loss: 3.5, perplexity: 32.6
- 綠色: block_size: 4096, avg loss: 3.44, perplexity: 31.5

![[Pasted image 20250108212311.png]]

Phi3-4K-mini (3.8B) on Wiki2, PTB, and Shakespeare datasets!
![[Pasted image 20250108144910.png]]

Llama3B 和 Phi3-3.8B on Wiki2, PTB, and Shakespeare datasets!
![[Pasted image 20250108211213.png]]



完整的 (6) code 如下

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt

# 定義參數
model_name = "gpt2"
dataset_name = "wikitext"
max_samples = None  # 如果需要限制處理的樣本數，設置此參數
block_size = 1024  # 每個樣本的 token 長度
stride = 512  # overlap 的 token 長度
batch_size = 4

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# 加載數據集
dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split="test")

# 將數據集的所有文本拼接為一個長字符串
text = " ".join(dataset["text"])
# 注意這裡 tokenizer 沒有限制最大 tokenization 長度。
tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]

# 構建樣本塊
samples = []

if stride == 0:
    stride = block_size  # Set stride to block_size if stride is zero
    
for i in range(0, tokens.size(1) - block_size + 1, stride):
    samples.append(tokens[:, i : i + block_size])

samples_tensor = torch.cat(samples, dim=0)  # 合併所有樣本為一個 tensor

# 構建 TensorDataset 和 DataLoader
dataset = TensorDataset(samples_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 計算損失和困惑度
def evaluate_model(max_samples=None):
    losses = []

    print("Evaluating...")
    for batch_idx, batch in enumerate(dataloader):
        inputs = batch[0].to(device)

        with torch.no_grad():
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            losses.append(loss.item())

        print(f"Batch {batch_idx + 1}: Loss = {loss.item():.4f}", flush=True)

        if max_samples and (batch_idx + 1) * batch_size >= max_samples:
            break

    average_loss = sum(losses) / len(losses) if losses else 0
    perplexity = torch.exp(torch.tensor(average_loss)).item() if losses else float('inf')
    return losses, average_loss, perplexity

# 評估模型並繪製損失曲線
losses, average_loss, perplexity = evaluate_model()

# 打印平均損失和困惑度
print(f"Average Loss: {average_loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")

# 繪製損失曲線
plt.figure(figsize=(10, 5))
plt.plot(losses, label="Loss")
plt.xlabel("Batch Index")
plt.ylabel("Loss")
plt.title("Loss Per Batch")
plt.legend()
plt.grid(True)
plt.show()
```

這是一種常見處理文本的方法，用於確保在基於上下文的模型（如 GPT）訓練或評估中保持上下文連貫性。以下是具體的步驟來實現這樣的過程：

### 方法：將所有記錄串接、切分並滑動窗口

1. **資料拼接**：
    
    - 將 `wikitext2` 的所有句子或記錄串接為一個完整的文本。
2. **Tokenize**：
    
    - 使用 `tokenizer` 將完整的文本轉換為 token。
3. **切片**：
    
    - 將 token 按照 `1024` 為單位切分。
    - 設定滑動窗口 `stride=512`，確保切分的區塊之間有一部分重疊。
4. **準備模型輸入**：
    
    - 每個區塊可以作為一個訓練或評估樣本，模型可計算這些樣本的 loss 和 perplexity。

以下是實現這一流程的範例代碼：

```python
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

# 加載 tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加載 Wikitext-2 測試集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# 拼接所有句子
text = " ".join(dataset["text"])

# Tokenize 整個文本
tokens = tokenizer(text, return_tensors="pt")["input_ids"]

# 定義切片長度和 stride
block_size = 1024
stride = 512

# 生成滑動窗口樣本
num_tokens = tokens.size(1)
samples = []
for i in range(0, num_tokens - block_size + 1, stride):
    samples.append(tokens[:, i:i + block_size])

print(f"生成了 {len(samples)} 個樣本，每個樣本的長度為 {block_size} tokens。")

# 準備模型輸入
samples_tensor = torch.cat(samples, dim=0)
print(samples_tensor.shape)
```

### 優點

- **上下文連續性**：通過滑動窗口確保重疊，模型可以利用之前區塊的上下文。
- **適用於長文本**：這種方式適合於處理長文本，避免模型因截斷而丟失上下文。

### 注意事項

- **計算成本**：由於滑動窗口產生了更多樣本，會增加計算成本。
- **重疊問題**：同一部分 token 可能會多次參與 loss 計算，需要在計算 perplexity 時適當處理。

以下是將 `samples_tensor` 按照 `batch_size=4` 打包的過程，並將其準備為模型的輸入：

### 代碼範例

```python
from torch.utils.data import DataLoader, TensorDataset

# 定義 batch size
batch_size = 4

# 構建 TensorDataset
dataset = TensorDataset(samples_tensor)

# 創建 DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 查看數據形狀
for batch_idx, batch in enumerate(dataloader):
    inputs = batch[0]  # 每個 batch 的 inputs
    print(f"Batch {batch_idx + 1}: {inputs.shape}")  # 應該是 (batch_size, block_size)
    # 假設只打印前三個 batch
    if batch_idx == 2:
        break
```

### 解釋

1. **`TensorDataset`**:
    
    - 將 `samples_tensor` 封裝為 PyTorch 的 TensorDataset。
2. **`DataLoader`**:
    
    - 利用 `batch_size` 和 `shuffle` 將樣本打包為批次，`shuffle=False` 表示保留樣本順序。
3. **迭代批次**:
    
    - 遍歷 `dataloader`，每次獲取一個批次。
    - 每個批次的形狀應為 `(batch_size, block_size)`。

### 數據輸入到模型

您可以直接將這些批次作為模型的輸入，例如：

```python
for batch in dataloader:
    inputs = batch[0].to(device)  # 移動到 GPU 或 CPU
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        print(f"Loss: {loss.item()}")
```

### 優化

1. **`DataLoader` 的使用**:
    
    - 確保內存效率，避免一次性加載過多數據。
2. **模型計算**:
    
    - 在 GPU 訓練時，批量處理有助於提升吞吐量。


```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt

# 定義參數
model_name = "gpt2"
dataset_name = "wikitext"
max_samples = None  # 如果需要限制處理的樣本數，設置此參數
block_size = 1024  # 每個樣本的 token 長度
stride = 512  # overlap 的 token 長度
batch_size = 4

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# 加載數據集
dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split="test")

# 將數據集的所有文本拼接為一個長字符串
text = " ".join(dataset["text"])
tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]

# 構建樣本塊
samples = []
for i in range(0, tokens.size(1) - block_size + 1, stride):
    samples.append(tokens[:, i : i + block_size])

samples_tensor = torch.cat(samples, dim=0)  # 合併所有樣本為一個 tensor

# 構建 TensorDataset 和 DataLoader
dataset = TensorDataset(samples_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 計算損失
def evaluate_model():
    losses = []

    print("Evaluating...")
    for batch_idx, batch in enumerate(dataloader):
        inputs = batch[0].to(device)

        with torch.no_grad():
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            losses.append(loss.item())

        print(f"Batch {batch_idx + 1}: Loss = {loss.item():.4f}", flush=True)

        if max_samples and (batch_idx + 1) * batch_size >= max_samples:
            break

    return losses

# 評估模型並繪製損失曲線
losses = evaluate_model()

# 繪製損失曲線
plt.figure(figsize=(10, 5))
plt.plot(losses, label="Loss")
plt.xlabel("Batch Index")
plt.ylabel("Loss")
plt.title("Loss Per Batch")
plt.legend()
plt.grid(True)
plt.show()

```









```python
def compute_ppl(eval_preds):
    """
    Calculate perplexity from evaluation predictions.

    Args:
        eval_preds (tuple): A tuple of logits and labels.

    Returns:
        dict: A dictionary containing the calculated perplexity.
    """
    # Debug message
    print(f'wy debug ppl--------------------------------')
    
    # Unpack logits and labels
    logits, labels = eval_preds
    print('logits', logits.shape, 'labels', labels.shape)
    
    # Convert logits and labels to PyTorch tensors
    logits = torch.from_numpy(logits)  # logits shape (num_example, seq, vocab_size)
    labels = torch.from_numpy(labels)  # labels shape (num_example, seq)
    
    num_example = labels.size(0)
    
    # Initialize the loss function
    loss_fct = CrossEntropyLoss(reduction='none')
    
    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten logits and labels
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Compute loss
    loss = loss_fct(shift_logits, shift_labels)
    print('output loss shape', loss.shape)
    
    # Reshape the loss to match the number of examples
    loss = loss.reshape(num_example, -1)
    print('reshaped loss shape', loss.shape)
    
    # Compute perplexity values
    ppl_values = torch.exp(torch.mean(loss, dim=-1))
    mean_ppl = torch.mean(ppl_values)
    
    # Return the result as a dictionary
    result = {"perplexity": float(mean_ppl)}
    return result

```

How to write a good paper:
https://www.youtube.com/watch?v=1hEI_yIkIl0&ab_channel=Tunadorable





## Source ChatGPT

計算 perplexity, 一般用 WikiText-2 因為比較小，品質比較好。

### **Comparison Table**

|**Feature**|**WikiText-2**|**WikiText-103**|**enwik8**|
|---|---|---|---|
|**Size**|~2M tokens|~103M tokens|~100M characters|
|**Vocabulary Size**|~33,000 tokens|~267,000 tokens|N/A (raw character-level)|
|**Preprocessing**|Minimal|Minimal|None (includes raw text)|
|**Task Focus**|Word-level modeling|Word-level modeling|Character-level modeling|
|**Use Cases**|Small-scale experiments|Large-scale pretraining|Byte/character-level tasks|
|**Computational Cost**|Low|High|Moderate|



The **WikiText-2**, **WikiText-103**, and **enwik8** datasets are commonly used for training and evaluating language models. Here's a detailed comparison of their differences:

---

### **1. WikiText-2**

- **Description**:
    
    - A smaller version of the WikiText-103 dataset.
    - Contains high-quality, clean English text extracted from Wikipedia articles.
- **Characteristics**:
    
    - **Size**:
        - Training: ~2 million tokens
        - Validation: ~217,000 tokens
        - Test: ~245,000 tokens
    - **Vocabulary**: ~33,000 unique tokens.
    - Designed for quick experimentation and smaller-scale model development.
- **Use Case**:
    
    - Useful for testing new language model architectures or techniques without requiring significant computational resources.
- **Trade-offs**:
    
    - Smaller corpus limits its usefulness for pretraining large language models.
    - Overfitting is a concern for larger models.

---

### **2. WikiText-103**

- **Description**:
    
    - A larger, more comprehensive version of the WikiText-2 dataset.
    - Extracted from English Wikipedia, with minimal preprocessing to maintain the natural structure of sentences and paragraphs.
- **Characteristics**:
    
    - **Size**:
        - Training: ~103 million tokens
        - Validation: ~217,000 tokens
        - Test: ~245,000 tokens
    - **Vocabulary**: ~267,000 unique tokens.
    - Retains long-term dependencies by preserving full article structure.
    - Includes rare and less frequent words due to its larger size.
- **Use Case**:
    
    - Suitable for training larger language models.
    - Useful for evaluating long-range dependency handling in language models.
- **Trade-offs**:
    
    - Requires more computational resources compared to WikiText-2.
    - Slower for rapid prototyping.

---

### **3. enwik8**

- **Description**:
    
    - A dataset derived from the first 100 million characters of an English Wikipedia XML dump.
    - Focuses on character-level language modeling rather than token-based processing.
- **Characteristics**:
    
    - **Size**:
        - Training: ~90 million characters
        - Validation: ~5 million characters
        - Test: ~5 million characters
    - Processed as raw text, meaning punctuation, HTML tags, and special characters are included.
    - Designed for character-level tasks, unlike WikiText which is word-level.
- **Use Case**:
    
    - Character-level language model research and compression algorithms.
    - Ideal for exploring models with byte-level representations or subword tokenization.
- **Trade-offs**:
    
    - Requires more steps to tokenize and preprocess compared to WikiText datasets.
    - May not be as suitable for word-level language modeling tasks.

---


---

### **Summary**

- **WikiText-2**: Best for quick experiments and smaller models.
- **WikiText-103**: Preferred for pretraining or evaluating word-level models on long-range dependencies.
- **enwik8**: Ideal for character-level tasks or byte-level processing research.


Here are snippets from the datasets **WikiText-2**, **WikiText-103**, and **Enwik8**. These snippets are extracted based on the general characteristics of the datasets:

---

### **WikiText-2**

- **Format**: Plain text, tokenized with spaces.
- **Style**: Contains diverse topics, but focuses on structured sentences with proper grammar.

```text
 = Valkyria Chronicles III = 

 Valkyria Chronicles III: Unrecorded Chronicles is a tactical role-playing video game developed by Sega and released for the PlayStation Portable in Japan. 
```

---

### **WikiText-103**

- **Format**: Plain text, similar to WikiText-2 but much larger in size (over 100 million tokens).
- **Style**: Same format and structure as WikiText-2 but covering a much broader range of topics and depth.

```text
 = Natural Language Processing = 

 Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.
```

---

### **Enwik8**

- **Format**: Raw ASCII text (character-level).
- **Style**: Includes Wikipedia content but focuses on unprocessed text (e.g., no tokenization, retains all special characters, and formatted as characters).

```text
<text xml:space="preserve" bytes="473000">British Isles

The British Isles are a group of islands off the north-western coast of continental Europe that include the islands of Great Britain, Ireland and over six thousand smaller isles. There are two sovereign states located on the islands: the United Kingdom of Great Britain and Northern Ireland, and Ireland. The British Isles also include the Crown Dependencies of the Isle of Man and the Channel Islands (which are considered part of the British Isles by the UK government).

</text>
```

---

### **Key Observations**

1. **Tokenization**:
    
    - WikiText-2 and WikiText-103: Tokenized at the word level, stored as plain text.
    - Enwik8: Character-level, raw ASCII format.
2. **Structure**:
    
    - WikiText datasets are relatively clean and focus on proper Wikipedia articles.
    - Enwik8 is raw, retains formatting and metadata like `<text>` tags.
3. **Purpose**:
    
    - WikiText-2: Suitable for quick experiments and small-scale language modeling.
    - WikiText-103: Large-scale language modeling with diverse and deep content.
    - Enwik8: Focuses on character-level language modeling.

---

Let me know if you'd like a code snippet to directly view these datasets in Python.




## Appendix

### Hugging Face Perplexity Example

使用 GPT-2  ($n_{ctx} = 1024$) 計算 perplexity.  

這裏使用 Hugging face 的 GPT2 和 WikiText dataset.  要事先 install 以下 packages.

```
pip install transformer
pip install datasets
```



```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda"
model_id = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
```

我們用 WikiText-2 dataset 評估 PPL.  

```python
from datasets import load_dataset

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
```



```python
import torch
from tqdm import tqdm

max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
```

最後 ppl = 16.45.

