---
title: 大(語言)模型推理效能
date: 2024-10-27 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-08-14-LLM_Adapting]]"
categories:
  - LLM
tags:
  - AI
  - LLM
  - Transformer
  - prompt
---




## Source

https://arxiv.org/pdf/2404.14294 A Survey on Efficient Inference for Large Language Models

https://arxiv.org/pdf/2006.16236 Efficient Transformers: A Survey 2022




## Training and Inference 兩階段

1.  Memory:  attention matrix A = softmax (Q K') :   LxL  :  **内存和 context length 是平方關係**。内存 size (= weight 40%,  attention matrix 30%) 決定一個 GPU 的上限。
2. Computation:  x (lxd) [Wq, Wk, Wv] (dx3d) and Q K' (lxl)  and softmax (LxL) : 在乘以 batch.  所以是 computation bound >> memory BW bound!!  **計算量也是和 context length 平方關係。**


## Inference 再分成兩階段：Prefill (prompt) and Decode (generation) 

Prefill = Training 
因爲 prefill mode 是 one-shot.   Prefill mode 的重點是 TTFT (Time-To-First-Token).  Dominate factor 主要是算力。但如果是 long input prompt, 內存也是重點。**但是因為是 causal mask, 可以一段一段填滿 kv cache 和 Attention matrix！**

整**體的 inference 時間主要是以 decode (generation) mode 爲主**。因為是 auto-regression token-by-token, **最大的瓶頸是在 kv cache + weight 的 memory BW.**

Generation mode:
1.  Memory:  attention matrix A = softmax (Q K') :  1xL.  KV Cache 代表**内存和 context length 是線性關係**。
2. Computation:  x (1xd) [Wq, Wk, Wv] (dx3d) and Q K' (1xL)  and softmax (1xL) : 在乘以 batch.  所以是 computation bound << memory BW bound!!  **計算量也是和 context length 線性關係。**

### Attention 部分比較表

| Stage                                     | Memory Requirement                              | Memory Bandwidth                    | Computation Requirement                   | Primary Bottleneck                                              | RNN                                |
| ----------------------------------------- | ----------------------------------------------- | ----------------------------------- | ----------------------------------------- | --------------------------------------------------------------- | ---------------------------------- |
| **Training**                              | Attn matrix with $L^2$                          | Weight only, shared by token length | Attn matrix grows with $L^2$              | **Computation-bound** (heavy matrix multiplications), 但可平行處理加速！ | Grows with $L$， Recursive, 無法平行加速！ |
| **Inference Prefill (one-shot training)** | same as above, 但可分段 to build causal attn matrix | same as above                       | Attn matrix grows with $L^2$, impact TTFT | **Computation-bound**, but **Memory-bound** for long prompts    | Grows with $L$                     |
| **Inference Generation (AR)**             | KV cache grows with $L$                         | Weight + KV cache for every token!  | Attn matrix grows with $L$                | **Memory bandwidth-bound** due to KV cache and weights          | Constant                           |


<img src="/media/image-20241031120636.png" alt="20241031120636" style="zoom:60%;" />

<img src="/media/image-20241027201735.png" alt="20241027201735" style="zoom:60%;" />


<img src="/media/image-20241027201707.png" alt="20241027201707" style="zoom:60%;" />



<img src="/media/image-20241027204323.png" alt="20241027204323" style="zoom:60%;" />


```mermaid
graph LR
    A[Inference] --> B[Data Opt]
    A --> C[Model Opt]
    A --> D[System Opt]

    B --> B1[Input Compression]
    B --> B2[Output Organization]

    B1 --> B11[Prompt Pruning]
    B1 --> B12[Prompt Summary]
    B1 --> B13[Soft Prompt Compression]
    B1 --> B14[RAG]

    C --> C1[Efficient Structure]
    C --> C2[Compression]

    C1 --> C11[Efficient FFN]
    C1 --> C12[Efficient Attention]
    C1 --> C13[Xformer Alternate]

    C2 --> C21[Quant]
    C2 --> C22[Sparse]
    C2 --> C23[Structure Opt]
    C2 --> C24[Distillation]
    C2 --> C25[Dynamic Inference]

    C21 --> C211[PTQ]
    C21 --> C212[QAT]

    C22 --> C221[Weight Pruning]
    C22 --> C222[Sparse Attention]

    C23 --> C231[Structure Factorization]
    C23 --> C232[NAS]

    C24 --> C241[White-box KD]
    C24 --> C242[Black-box KD]

    D --> D1[Inference Engine]
    D --> D2[Serving System]

    D1 --> D11[Graph and Operator Opt]
    D1 --> D12[Offloading]
    D1 --> D13[Speculative Decoding]
    D1 --> D14[Memory Management]

    D2 --> D21[Batching]
    D2 --> D22[Scheduling]
    D2 --> D23[Distributed Systems]
```




