---
title: 文本分類 - IMDB 意見分析
date: 2024-05-01 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
typora-root-url: ../../allenlu2009.github.io


---





## Source

[pytorch-sentiment-analysis/1 - Neural Bag of Words.ipynb at main · bentrevett/pytorch-sentiment-analysis (github.com)](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/main/1 - Neural Bag of Words.ipynb)  Excellent example!

[trinhxuankhai/mamba_text_classification · Hugging Face](https://huggingface.co/trinhxuankhai/mamba_text_classification)  : Mamba text classification

[LLMs-from-scratch/ch06/01_main-chapter-code/ch06.ipynb at main · rasbt/LLMs-from-scratch (github.com)](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/ch06.ipynb):  SPAM detection.





## 自然語言處理（NLP）：文本分類

在大語言模型 (LLM) 出現之前，分析式 (監督式) 自然語言處理是主流。其中文本分類是最常見的應用。

反過來是用 LLM 來做 NLP 工作，例如文本分類。

我們遵循煉丹五部曲：

0. 應用 (起死回生，長生不老，美顔)：analytic (spam, sentiment) or generative (summarization)

1. 靈材：datasets, tokenizer, embedding

2. 丹方：model

3. 丹爐：Nvidia GPU (和財力有關)

4. 煉製：training:  data_loader, loss function, optimizer

5. 評估：evaluation:  accuracy,  BLEU



### 應用

NLP 的應用基本是分析式（監督式）AI.   不過我們可以用生成式 AI **微調**作爲分析式 AI.  我們會用 SCAM detection 爲例。

#### IMDB 意見分析

是基本例子。可以參考 reference.  我們就直接列出結果

|                           | Tokenizer    | Embedding | Train loss | Val loss | Train Accuracy | Val Accuracy |
| ------------------------- | ------------ | --------- | ---------- | -------- | -------------- | ------------ |
| NBOW (Neural Bag of Word) | Spacy? space | GloVe     | 0.28       | 0.32     | 91%            | 88%          |
| RNN                       |              |           | 0.22       | 0.34     | 91%            | 88%          |
| CNN                       |              |           | 0.12       | 0.28     | 96%            | 89%          |
| Transformer BERT - 109M   |              |           | 0.07       | 0.25     | **98%**        | **93%**      |
| Transformer - 30M         |              |           | 0.21       | 0.39     | 92%            | 87%          |
| Mamba - 130M              | HF - Auto    |           | **0.0006** | **0.17** | 94%            | **94%**      |



#### Tokenizer/Embedding

* GloVe:  pre-train 





#### 應用: SPAM 分析：見 PEFT 分析。







#### 應用: 語音詐騙分析
