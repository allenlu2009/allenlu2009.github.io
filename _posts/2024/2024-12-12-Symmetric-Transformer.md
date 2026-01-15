---
title: Symmetric Transformer
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

Symmetric transformer 的重點是 symmetric attention, K = Q!

**Symmetric transformer 最大的好處是 low-rank!  對於 long-context 有致命吸引力。**

因為 $Q K' = Q Q'$.    $Q = x W_q$ (l x d) 在 long-context 本來就是 low-rank.
$A = Q Q'$   rank <=d， $M = Q' Q$  的 rank 和 A 一樣
$A^2 = Q M Q'$ 的 rank 和 A 一樣
$A^k = Q M^{k-1} Q'$  一樣。以及所有 f(A) in polynomial 或是 exp(A) (注意這是 matrix exponential, 不是 element-wise exponential!). 都是 low-rank.

exp-element-wise(A) 不是直接 low-rank, 可能是 empirical low rank.  可以用一些例子來看：

https://www.youtube.com/watch?v=-_2AF9Lhweo

一個 Q K' 的 low-rank (rank=128)
![[Pasted image 20241219211602.png]]

做完 softmax (or element-wise exponential), 不是那麼 low-rank (~300)
![[Pasted image 20241219211714.png]]


## Case 1:  使用 M : causal mask for LLM generation purpose!

不過我試了一下，無法用 f(Q Q')得到比較好的結果。還是只能用 QQ'  row-based 的 operations,  也就是 $q_i \cdot q_j$  為 base 的 operation.

#### Work:

1. Norm:  $A = 1 + \frac{q_i \cdot q_j}{\vert q_i \vert \vert q_j\vert}$   Shakespear loss: 1.1   
2. Softmax:  $A = softmax(q_i \cdot q_j) = softmax(Q Q')$.  Asymmetric KQ softmax: 3L:0.69 / 6L:0.41, symmetric softmax: 0.84.
3. Differential softmax:  $A = softmax(Q_1 Q_1') - \lambda softmax(Q_2, Q_2')$.   Asymmetric KQ differential softmax: 3L:0.69, symmetric differential softmax 3L:0.69 / 6L:0.48.  
	1. 也就是說 differential 對 asymmetric loss 沒有幫助，0.69->0.69。但對 symmetric loss 有幫助。0.84 -> 0.69 


因為 train 是用 model.train().  validate/test 是 model.eval()。差異應該是 dropout? 因為 validate 沒有 dropout (model.eval(), 所以 validate 的 loss 比較小！
把 learning rate 從 0.0003 變成 0.00003 可以讓最後的 loss (<0.2) 變得更小。

|                   | Nanogpt   | gpt+Diff  | gpt+Sym  | gpt+Diff+Sym | Norm+Sym  |
| ----------------- | --------- | --------- | -------- | ------------ | --------- |
| 3L,train/validate | 0.69      | 0.69      | 0.84     | 0.69         | 1.1       |
| 6L,train/validate | 0.18/0.08 | 0.21/0.09 | 0.5/0.22 | 0.23/0.1     | 0.94/0.77 |
| 8L,train/validate | 0.29/0.11 | 0.14/0.08 |          | 0.21/0.1     |           |

#### Not Work:

1. Matrix exponential:  $\exp(Q Q')$ :  即使加上各種 normalization, l2 norm, row-based nom.
2. 簡單 norm:  1 + $\frac{Q Q'}{\vert Q Q' \vert ^2}$



Differential Transformer + Symmetry 似乎有比較好的 noise cancelling effect?  因為 $\lambda \approx 1$

nanogpt differential  6 layers  $\lambda_{full}$  = 0.23, 0.76, 0.68, 0.69, 0.76, 0.78, 
nanogpt differential symmetry 6 layer $\lambda_{full}$  = 0.45, 0.82, 0.82, 0.80, 0.85, 0.94
Here's the updated table with the sequence changed to $\lambda_1$, $\lambda_2$, $\lambda_{full}$:

|           | **NanoGPT** | Differential    |                      |                      | **NanoGPT** | Differential    | Symmetry             |                      |
| --------- | ----------- | --------------- | -------------------- | -------------------- | ----------- | --------------- | -------------------- | -------------------- |
| Parameter | $\lambda_1$ | **$\lambda_2$** | **$\lambda_{init}$** | **$\lambda_{full}$** | $\lambda_1$ | **$\lambda_2$** | **$\lambda_{init}$** | **$\lambda_{full}$** |
| Layer 1   | 0.98        | 1.04            | 0.36                 | 0.30                 | 1.05        | 0.87            | 0.36                 | 0.53                 |
| Layer 2   | 1.14        | 0.84            | 0.47                 | 0.77                 | 1.15        | 0.80            | 0.47                 | 0.83                 |
| Layer 3   | 1.02        | 0.90            | 0.56                 | 0.68                 | 1.10        | 0.84            | 0.56                 | 0.82                 |
| Layer 4   | 0.99        | 0.91            | 0.62                 | 0.69                 | 1.11        | 0.93            | 0.62                 | 0.80                 |
| Layer 5   | 1.05        | 0.96            | 0.67                 | 0.76                 | 1.08        | 0.89            | 0.67                 | 0.85                 |
| Layer 6   | 1.02        | 0.94            | 0.7                  | 0.78                 | 1.05        | 0.80            | 0.7                  | 0.95                 |
| Loss      |             |                 |                      | 0.22                 |             |                 |                      | 0.24                 |

Let me know if you want me to make further adjustments!


## Case 2:  不使用 M : for LLM classification (BERT) purpose
## Case 3: 不使用 M: for image purpose



**Symmetric transformer 最大的問題是表現力。是否會 miss 一些 asymmetric transformer 能抓到的細節。**

可能的解法就是用 function f(A) = f(Q Q') 增加表現力。

原來的 transformer 的問題:
- asymmetric :  K Q'
- 原始的 transformer 是用 element-wise exponential 提供非線性和保持正性 > 0:   但是 element-wise exponential 也會破壞 low-rank!
	- symmetric matrix (QQ'). 是 PSD，以及 f(QQ') 是否可以保證 > 0.

注意這裡的 f(QQ') 是以 matrix 為單位，所以不含 element-wise operation, like exp.   如果是 element-wise operation like causal mask 或是 exp(),  應該還是有一定的 low-rank?  只是要用 random feature 方法來近似？



**exp(A) 的幾個問題： 這裡是 matrix exponential, 不是 element-wise exponential**
1. 如何做 generation, 也就是 iterative or incremental update for causal case?  好像沒有辦法不改變之前的結果？
2. 如何保證像 softmax normalization 也就是每個 row 都是正值？可以用 Markovian equation 來看 transitional probability?


以下 from perplexity

The performance of using symmetric attention by setting $$ K = Q $$ in self-attention mechanisms can lead to several implications, both positive and negative. Here are the key points regarding this approach:

## Implications of Symmetric Attention

1. **Loss of Asymmetry**: 
   - The original self-attention mechanism is designed to be asymmetric, meaning that the attention weights are computed based on different representations of queries and keys. This asymmetry allows for nuanced interactions where different contexts can influence how attention is distributed. By making $$ K = Q $$, this nuanced interaction may be lost, potentially leading to less informative attention distributions[1][4].

2. **Potential for Symmetric Relationships**:
   - In some contexts, symmetric attention could simplify the model and make it easier to interpret. For instance, if the relationships between inputs are inherently symmetric, using $$ K = Q $$ could yield more intuitive results where the attention reflects mutual influence equally[1][2].

3. **Computational Efficiency**:
   - Using symmetric attention can reduce computational complexity since the same matrix is used for both queries and keys. This could lead to faster computations in scenarios where efficiency is critical, although it might come at the cost of expressiveness and performance in capturing complex dependencies[3][4].

4. **Empirical Performance**:
   - While theoretical considerations suggest that symmetric attention may not capture all relationships as effectively as asymmetric attention, empirical studies are necessary to validate these claims. Some research indicates that models using asymmetric kernels (like those derived from Kernel SVD) may achieve better performance due to their ability to represent more complex relationships within the data[2][4].

5. **Low-Rank Properties**:
   - The attention matrices in Transformers often exhibit low-rank properties, especially in deeper layers. Using symmetric attention could affect how these low-rank characteristics are exploited, potentially limiting the model's ability to generalize well across tasks[2][4].

## Conclusion

In summary, while using $$ K = Q $$ for symmetric attention may offer benefits such as computational efficiency and simplicity, it risks losing the rich interactions that asymmetric attention provides. The effectiveness of this approach would ultimately depend on the specific application and context, necessitating further empirical validation to assess its impact on performance compared to traditional asymmetric self-attention methods.


```
=============================================================================
Layer (type:depth-idx)                             Output Shape              Param #
=============================================================================
CursorGPT                                     [32, 300, 65]             --
├─Embedding: 1-1                              [32, 300, 256]            16,640
├─Embedding: 1-2                              [32, 300, 256]            76,800
├─Dropout: 1-3                                [32, 300, 256]            --
├─ModuleList: 1-4                             --                        --
│    └─CursorGPTLayer: 2-1                    [32, 300, 256]            --
│    │    └─LayerNorm: 3-1                    [32, 300, 256]            512
│    │    └─CustomMultiheadAttention: 3-2     [32, 300, 256]            --
│    │    │    └─Linear: 4-1                  [32, 300, 256]            65,792
│    │    │    └─Linear: 4-2                  [32, 300, 256]       (recursive)
│    │    │    └─Linear: 4-3                  [32, 300, 256]            65,792
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
│    │    └─CustomMultiheadAttention: 3-8     [32, 300, 256]            --
│    │    │    └─Linear: 4-8                  [32, 300, 256]            65,792
│    │    │    └─Linear: 4-9                  [32, 300, 256]       (recursive)
│    │    │    └─Linear: 4-10                 [32, 300, 256]            65,792
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
│    │    └─CustomMultiheadAttention: 3-14    [32, 300, 256]            --
│    │    │    └─Linear: 4-15                 [32, 300, 256]            65,792
│    │    │    └─Linear: 4-16                 [32, 300, 256]       (recursive)
│    │    │    └─Linear: 4-17                 [32, 300, 256]            65,792
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
=============================================================================
Total params: 2,282,561
Trainable params: 2,282,561
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 79.36
=============================================================================
Input size (MB): 0.08
Forward/backward pass size (MB): 712.78
Params size (MB): 9.13
Estimated Total Size (MB): 721.99
=============================================================================

```



Citations:
[1] https://stackoverflow.com/questions/75772288/how-to-read-a-bert-attention-weight-matrix
[2] https://proceedings.neurips.cc/paper_files/paper/2023/file/cd687a58a13b673eea3fc1b2e4944cf7-Paper-Conference.pdf
[3] https://arxiv.org/html/2402.17507v1
[4] https://openreview.net/forum?id=bRyduWAAVT
[5] https://ieeexplore.ieee.org/document/10380980/




Specifi. ally, given input $X \in \mathbb{R}^{N \times d_{\text {model }}}$, we first project them to query, key, and value $Q_1, Q_2, K_1, K_2 \in \mathbb{R}^{N \times d}, V \in \mathbb{R}^{N \times 2 d}$. Then the differential attention operator DiffAttn $(\cdot)$ computes outputs via:

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


