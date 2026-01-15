---
title: Differential Transformer
date: 2024-12-08 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-08-14-LLM_Adapting]]"
categories:
  - LLM
tags:
  - Attention
  - LLM
  - Transformer
---


## Source

Orig differential transformer paper: https://arxiv.org/pdf/2410.05258


## Overview

Differential transformer 的重點是 differential attention.

Specifically, given input $X \in \mathbb{R}^{N \times d_{\text {model }}}$, we first project them to query, key, and value $Q_1, Q_2, K_1, K_2 \in \mathbb{R}^{N \times d}, V \in \mathbb{R}^{N \times 2 d}$. Then the differential attention operator DiffAttn $(\cdot)$ computes outputs via:

$$
\begin{gathered}
{\left[Q_1 ; Q_2\right]=X W^Q, \quad\left[K_1 ; K_2\right]=X W^K, \quad V=X W^V} \\
\operatorname{DiffAttn}(X)=\left(\operatorname{softmax}\left(\frac{Q_1 K_1^T}{\sqrt{d}}\right)-\lambda \operatorname{softmax}\left(\frac{Q_2 K_2^T}{\sqrt{d}}\right)\right) V
\end{gathered}
$$

where $W^Q, W^K, W^V \in \mathbb{R}^{d_{\text {model }} \times 2 d}$ are parameters, and $\lambda$ is a learnable scalar. 

看起來好像 double Q and K,  實際上會把 head_size 減半，以保持整體參數量。

![[Pasted image 20241208145525.png]]

不過還有一個 **learnable 參數是 $\lambda$**.  其實這并不是一個參數，而是多個參數
- 每一層 (layer) 都有自己的 $\lambda$，也就是有多少 layer, 就有多少 $\lambda$.   
- 每一層 $\lambda$ 如何得到？可以看 code 如下。
$$
\lambda=\exp \left(\lambda_{\mathbf{q}_1} \cdot \lambda_{\mathbf{k}_1}\right)-\exp \left(\lambda_{\mathbf{q}_2} \cdot \lambda_{\mathbf{k}_2}\right)+\lambda_{\text {init }}
$$
- 其中 $\lambda_{q_i,k_i}$ 都是 head size,  就是每一個 head 有自己的 $\lambda$,  exp 內的乘法是內積，是個 scalar 衡量相似程度？最後的 $\lambda_{init}$ 也是一個 scalar 常數, 而且和 layer index 相關。非常奇怪？
	- $\lambda_{\text {init }}=0.8-0.6 \times \exp (-0.3 \cdot(l-1))$ where $l \in[1, L]$ represents layer index. 
- 所謂 $\lambda$ 是 learnable 應該指 $\lambda$  和  $\lambda_{q_i,k_i}$ 而非 $\lambda_{init}$ ! 

where $\lambda_{\mathbf{q}_1}, \lambda_{\mathbf{k}_1}, \lambda_{\mathbf{q}_2}, \lambda_{\mathbf{k}_{\mathbf{2}}} \in \mathbb{R}^d$ are learnable vectors, and $\lambda_{\text {init }} \in(0,1)$ is a constant used for the initialization of $\lambda$. We empirically find that the setting  works well in practice.  

但是在 learnable parameters 似乎沒有看到？ Yes:  基本多了 4 x (256/8/2) = 64.

- 因爲 attention 是相減，所以可能有負值
- 使用 GroupNorm for each head
- 最後 $O = O \cdot (1-\lambda_{init})$   主要是讓 back-prop 和原來的 transformer 一樣。  

![[Pasted image 20241208204055.png]]


## Advantages

1.  Differential Attention 最大的賣點是 noise cancellation in circuit and audio，增加 SNR.  同樣的邏輯也用於 attention, 特別是對於比較遠的 context.
2. 另一個可能的賣點是 Q，K 用於對稱而不是非對稱 attention.



## Code: Diff_NanoGPT

正常的 Differential Transformer

- Diff nanogpt model size: 2.480225M vs. 2.479,937M (多了 288=3x96=3x(64+32))
- Diff nanogpt post layer normalization: 512 = 256 x 2 (bias + gain)
- 因爲 $\lambda$，每一層多了 4 x (256/8/2) = 64 個參數
- 另外 RMSNorm 是 2 x (256/8/2) = 32 個參數
- Parameter size (MB) : 9.92MB = 2.5M x 4 byte (32-bit for long format)
- Training: input shape: [batch_size, block_size, vocab_size] = [32, 256, 65]
- Training: embed shape: [batch_size, block_size, n_embed] = [32, 256, 300]
- Training: loss is small: 0.7

```
=================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
=================================================================================
CursorGPT                                     [32, 300, 65]             --
├─Embedding: 1-1                              [32, 300, 256]            16,640
├─Embedding: 1-2                              [32, 300, 256]            76,800
├─Dropout: 1-3                                [32, 300, 256]            --
├─ModuleList: 1-4                             --                        --
│    └─CursorGPTLayer: 2-1                    [32, 300, 256]            --
│    │    └─LayerNorm: 3-1                    [32, 300, 256]            512
│    │    └─MultiheadDiffAttention: 3-2       [32, 300, 256]            64
│    │    │    └─Linear: 4-1                  [32, 300, 256]            65,792
│    │    │    └─Linear: 4-2                  [32, 300, 256]            65,792
│    │    │    └─Linear: 4-3                  [32, 300, 256]            65,792
│    │    │    └─RMSNorm: 4-4                 [32, 8, 300, 32]          32
│    │    │    └─Linear: 4-5                  [32, 300, 256]            65,792
│    │    └─Dropout: 3-3                      [32, 300, 256]            --
│    │    └─LayerNorm: 3-4                    [32, 300, 256]            512
│    │    └─Sequential: 3-5                   [32, 300, 256]            --
│    │    │    └─Linear: 4-6                  [32, 300, 1024]           263,168
│    │    │    └─ReLU: 4-7                    [32, 300, 1024]           --
│    │    │    └─Linear: 4-8                  [32, 300, 256]            262,400
│    │    └─Dropout: 3-6                      [32, 300, 256]            --
│    └─CursorGPTLayer: 2-2                    [32, 300, 256]            --
│    │    └─LayerNorm: 3-7                    [32, 300, 256]            512
│    │    └─MultiheadDiffAttention: 3-8       [32, 300, 256]            64
│    │    │    └─Linear: 4-9                  [32, 300, 256]            65,792
│    │    │    └─Linear: 4-10                 [32, 300, 256]            65,792
│    │    │    └─Linear: 4-11                 [32, 300, 256]            65,792
│    │    │    └─RMSNorm: 4-12                [32, 8, 300, 32]          32
│    │    │    └─Linear: 4-13                 [32, 300, 256]            65,792
│    │    └─Dropout: 3-9                      [32, 300, 256]            --
│    │    └─LayerNorm: 3-10                   [32, 300, 256]            512
│    │    └─Sequential: 3-11                  [32, 300, 256]            --
│    │    │    └─Linear: 4-14                 [32, 300, 1024]           263,168
│    │    │    └─ReLU: 4-15                   [32, 300, 1024]           --
│    │    │    └─Linear: 4-16                 [32, 300, 256]            262,400
│    │    └─Dropout: 3-12                     [32, 300, 256]            --
│    └─CursorGPTLayer: 2-3                    [32, 300, 256]            --
│    │    └─LayerNorm: 3-13                   [32, 300, 256]            512
│    │    └─MultiheadDiffAttention: 3-14      [32, 300, 256]            64
│    │    │    └─Linear: 4-17                 [32, 300, 256]            65,792
│    │    │    └─Linear: 4-18                 [32, 300, 256]            65,792
│    │    │    └─Linear: 4-19                 [32, 300, 256]            65,792
│    │    │    └─RMSNorm: 4-20                [32, 8, 300, 32]          32
│    │    │    └─Linear: 4-21                 [32, 300, 256]            65,792
│    │    └─Dropout: 3-15                     [32, 300, 256]            --
│    │    └─LayerNorm: 3-16                   [32, 300, 256]            512
│    │    └─Sequential: 3-17                  [32, 300, 256]            --
│    │    │    └─Linear: 4-22                 [32, 300, 1024]           263,168
│    │    │    └─ReLU: 4-23                   [32, 300, 1024]           --
│    │    │    └─Linear: 4-24                 [32, 300, 256]            262,400
│    │    └─Dropout: 3-18                     [32, 300, 256]            --
├─LayerNorm: 1-5                              [32, 300, 256]            512
├─Linear: 1-6                                 [32, 300, 65]             16,705
=================================================================================
Total params: 2,480,225
Trainable params: 2,480,225
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 79.36
=================================================================================
Input size (MB): 0.08
Forward/backward pass size (MB): 771.76
Params size (MB): 9.92
Estimated Total Size (MB): 781.76
================================================================================
```

Total params: 2,479,937



## Code: Diff_Symmetry

對稱的 Symmetry Differential Transformer

- Diff nanogpt model size: 2.282249M
- Diff nanogpt post layer normalization: 512 = 256 x 2 (bias + gain)
- 因爲 $\lambda$，每一層多了 4 x (256/8/2) = 64 個參數
- 另外 RMSNorm 是 2 x (256/8/2) = 32 個參數
- Parameter size (MB) : 9.1MB = 2.3M x 4 byte (32-bit for long format)
- Training: input shape: [batch_size, block_size, vocab_size] = [32, 256, 65]
- Training: embed shape: [batch_size, block_size, n_embed] = [32, 256, 300]
- Training: loss is small: 0.7

```
=================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
=================================================================================
CursorGPT                                     [32, 300, 65]             --
├─Embedding: 1-1                              [32, 300, 256]            16,640
├─Embedding: 1-2                              [32, 300, 256]            76,800
├─Dropout: 1-3                                [32, 300, 256]            --
├─ModuleList: 1-4                             --                        --
│    └─CursorGPTLayer: 2-1                    [32, 300, 256]            --
│    │    └─LayerNorm: 3-1                    [32, 300, 256]            512
│    │    └─MultiheadDiffAttention: 3-2       [32, 300, 256]            64
│    │    │    └─Linear: 4-1                  [32, 300, 256]            65,792
│    │    │    └─Linear: 4-2                  [32, 300, 256]            65,792
│    │    │    └─RMSNorm: 4-3                 [32, 8, 300, 32]          32
│    │    │    └─Linear: 4-4                  [32, 300, 256]            65,792
│    │    └─Dropout: 3-3                      [32, 300, 256]            --
│    │    └─LayerNorm: 3-4                    [32, 300, 256]            512
│    │    └─Sequential: 3-5                   [32, 300, 256]            --
│    │    │    └─Linear: 4-5                  [32, 300, 1024]           263,168
│    │    │    └─ReLU: 4-6                    [32, 300, 1024]           --
│    │    │    └─Linear: 4-7                  [32, 300, 256]            262,400
│    │    └─Dropout: 3-6                      [32, 300, 256]            --
│    └─CursorGPTLayer: 2-2                    [32, 300, 256]            --
│    │    └─LayerNorm: 3-7                    [32, 300, 256]            512
│    │    └─MultiheadDiffAttention: 3-8       [32, 300, 256]            64
│    │    │    └─Linear: 4-8                  [32, 300, 256]            65,792
│    │    │    └─Linear: 4-9                  [32, 300, 256]            65,792
│    │    │    └─RMSNorm: 4-10                [32, 8, 300, 32]          32
│    │    │    └─Linear: 4-11                 [32, 300, 256]            65,792
│    │    └─Dropout: 3-9                      [32, 300, 256]            --
│    │    └─LayerNorm: 3-10                   [32, 300, 256]            512
│    │    └─Sequential: 3-11                  [32, 300, 256]            --
│    │    │    └─Linear: 4-12                 [32, 300, 1024]           263,168
│    │    │    └─ReLU: 4-13                   [32, 300, 1024]           --
│    │    │    └─Linear: 4-14                 [32, 300, 256]            262,400
│    │    └─Dropout: 3-12                     [32, 300, 256]            --
│    └─CursorGPTLayer: 2-3                    [32, 300, 256]            --
│    │    └─LayerNorm: 3-13                   [32, 300, 256]            512
│    │    └─MultiheadDiffAttention: 3-14      [32, 300, 256]            64
│    │    │    └─Linear: 4-15                 [32, 300, 256]            65,792
│    │    │    └─Linear: 4-16                 [32, 300, 256]            65,792
│    │    │    └─RMSNorm: 4-17                [32, 8, 300, 32]          32
│    │    │    └─Linear: 4-18                 [32, 300, 256]            65,792
│    │    └─Dropout: 3-15                     [32, 300, 256]            --
│    │    └─LayerNorm: 3-16                   [32, 300, 256]            512
│    │    └─Sequential: 3-17                  [32, 300, 256]            --
│    │    │    └─Linear: 4-19                 [32, 300, 1024]           263,168
│    │    │    └─ReLU: 4-20                   [32, 300, 1024]           --
│    │    │    └─Linear: 4-21                 [32, 300, 256]            262,400
│    │    └─Dropout: 3-18                     [32, 300, 256]            --
├─LayerNorm: 1-5                              [32, 300, 256]            512
├─Linear: 1-6                                 [32, 300, 65]             16,705
=================================================================================
Total params: 2,282,849
Trainable params: 2,282,849
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 73.05
=================================================================================
Input size (MB): 0.08
Forward/backward pass size (MB): 712.78
Params size (MB): 9.13
Estimated Total Size (MB): 721.99
=================================================================================
```


