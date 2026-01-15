---
title: AI Evolution
date: 2024-08-15 23:10:08
description: LLM Output Token Rate
typora-root-url: ../../allenlu2009.github.io
categories:
  - Science
tags:
  - Boltzman_Machine
  - Hopfield
  - Ising
---





## Source

1. https://www.youtube.com/watch?v=_bqa_I5hNAo
2. Cross-Entropy loss, excellent video: https://youtu.be/KHVR587oW8I

## Takeaways

* Auto-encoder vs. Variational Auto-Encoder is like Hopfield network (memory or discrimanative) vs. Boltzman machine (generative part)?

* **Karpathy said neural network is only a compression machine.  The is only like Hopfield network.  But neural network has the generative part not captured by compression machine!!!!!**

- **Machine Learning = Compression (Hopfield network capability) + Generation (Stochasticity is the key + Hidden Units)**

- Analytic AI (discriminative AI): model focuses on cross-entropy loss.  Generative AI is variational inference: model focus on cross-entropy loss (or reconstruction loss) + regularization loss (KL divergence gap)

- Less is more structure!


## Introduction

It sounds like a big title.  That's not what I mean. I just need to come up with a better title later.
I am inspired by the youtube video.  **Specifically, how AI is evolved (or disrupt) from deterministic to stocahstic.**   This is the key.   **How about the other key, hidden states?**

Application:  procedure task (like autonomous drive, ZIP code detection)  --> (maybe a transition of sytle change) -->  creative task like song/article/music creation!

Analogy:   compression machine (in Karpathy's youtube video)  --> generative AI (auto-regression or diffusion)

Model:   Hopfield  network --> (Rristriced) Boltzman machine

Algorithm:  discriminative (encoder like) --> Generative (decoder based) [[2024-07-24-Less_Is_More_Transformer]]



<img src="/media/image-20240817101556.png" alt="20240817101556" style="zoom:60%;" />



## 重點：

* 自動編碼器與變分自動編碼器就像霍普菲爾德網絡（記憶或區分性）與玻爾茨曼機（生成部分）？

* 卡爾帕西表示神經網絡只是一個壓縮機。這僅類似於霍普菲爾德網絡。但神經網絡擁有壓縮機無法捕捉到的生成部分!!!!

## 介紹：

這聽起來像是一個大標題。這不是我所想的。我只是需要稍後想出一個更好的標題。
我受到YouTube視頻的啟發。具體來說，人工智慧是如何從確定性演變（或顛覆）為隨機性。

應用：程序任務（例如自主駕駛、郵遞區號檢測） --> （也許是一種風格轉變的過渡） --> 創意任務，如歌曲/文章/音樂創作！

模型：霍普菲爾德網絡 --> 玻爾茲曼機

算法：區分性（類似編碼器） --> 生成性（基於解碼器）[[2024-07-24-Less_Is_More_Transformer]]


