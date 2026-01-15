---
title: From RNN to Linear Attention to Mamba
date: 2025-03-24 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-1027-Attention_Kernel]]"
categories:
  - Science
tags:
  - Algorithm
  - Attention
  - Causal
  - Cumsum
  - Kernel
  - LLM
  - Transformer
---





## Source

1. (Very good！) Lee Hung-Yi YouTube: https://www.youtube.com/watch?v=gjsdVi90yQo&ab_channel=Hung-yiLee
2. Songlin Yang YouTube: https://www.youtube.com/watch?v=d0HJvGSWw8A&ab_channel=SashaRush
3. 

Paper: https://arxiv.org/pdf/2405.16605



Gated linear attention + Chunk!  Songlin Yang:  https://arxiv.org/pdf/2312.06635


![[Pasted image 20241120004206.png]]

## Block Comparison


![[Pasted image 20241212171142.png]]

## 數學

### Attention and Linear Attention

![[Pasted image 20241212171236.png]]


Non-autoregressive 的 V 可以寫成 matrix multiplication.  Autoregressive 的 V 只能寫成 element-wise multiplication with causal mask M.
![[Pasted image 20241205182440.png]]

#### Matrix multiplication form and element-wise matrix multiplication form of non-autoregressive vs. autoregressive


#### Mamba

![[Pasted image 20241212172546.png]]

The selective state space model formulated in eq. (9) can only deal with scalar input $x_i \in \mathbb{R}$. To operate over an input sequence $\boldsymbol{x} \in \mathbb{R}^{N \times C}, \boldsymbol{x}_i \in \mathbb{R}^{1 \times C}$, Mamba applies eq. (9) independently to each channel, leading to the following formulations:

$$
\begin{aligned}
\boldsymbol{h}_i & =\widetilde{\boldsymbol{A}}_i \odot \boldsymbol{h}_{i-1}+\boldsymbol{B}_i\left(\boldsymbol{\Delta}_i \odot \boldsymbol{x}_i\right), & & \boldsymbol{x}_i, \boldsymbol{\Delta}_i \in \mathbb{R}^{1 \times C}, \widetilde{\boldsymbol{A}}_i, \boldsymbol{h}_{i-1}, \boldsymbol{h}_i \in \mathbb{R}^{d \times C}, \quad \boldsymbol{B}_i \in \mathbb{R}^{d \times 1} \\
\boldsymbol{y}_i & =\boldsymbol{C}_i \boldsymbol{h}_i+\boldsymbol{D} \odot \boldsymbol{x}_i, & & \boldsymbol{y}_i \in \mathbb{R}^{1 \times C}, \boldsymbol{C}_i \in \mathbb{R}^{1 \times d}, \quad \boldsymbol{D} \in \mathbb{R}^{1 \times C},
\end{aligned}
$$

where $\boldsymbol{B}_i, \boldsymbol{C}_i, \boldsymbol{\Delta}_i$ are derived from the input. Specifically, Mamba employs $\boldsymbol{B}=\left(\boldsymbol{x} \mathbf{W}_B\right)^{\top}, \boldsymbol{C}=$ $\boldsymbol{x} \mathbf{W}_C, \boldsymbol{\Delta}=\operatorname{Softplus}\left(\boldsymbol{x} \mathbf{W}_1 \mathbf{W}_2\right)$ to produce the parameters $\boldsymbol{B} \in \mathbb{R}^{d \times N}, \boldsymbol{C} \in \mathbb{R}^{N \times d}, \boldsymbol{\Delta} \in \mathbb{R}^{N \times C}$, where $\mathbf{W}_B, \mathbf{W}_C \in \mathbb{R}^{C \times d}, \mathbf{W}_1 \in \mathbb{R}^{C \times C_0}, \mathbf{W}_2 \in \mathbb{R}^{C_0 \times C}$ are projection matrices. Notably, eq. (10) is exactly the selective SSM employed in Mamba, we only make modifications to formula formats.

## 兩者差異的詮釋

![[Pasted image 20241212173444.png]]
### 1.  Mamba 有 Input gate, $\Delta_i$!  Linear Attention 沒有。  
### 2. Mamba 有 forget gate，$A_i$ ! linear attention 沒有。是第一圖右邊上方紅色的部分

### 3. Mamba 有 shortcut, D, linear attention 沒有，wrong! 因為後面有就是第一圖右邊上方紅色的部分
### 3.  Mamba 有 shortcut, D,  linear attention 沒有。雖然後面都有 short cut.  不過 D 是 learnable parameter，而且是 element-wise 乘法。
### 4.  Linear attention 有 normalization,  Mamba 沒有


![[Pasted image 20241212171142.png]]

Linear Attention 可以用 (a) decay rate; (b) gated linear attention 補足

(a) 就是上圖右上的紅 box 是 $\lambda < 1$

A lower triangular matrix for a causal mask, $\mathbf{M}$, ensures that a token can only attend to itself and tokens before it, preventing future tokens from influencing the current one. Alternatively, this can be represented in matrix form for a sequence of length $t$:

$$
M =
\begin{bmatrix}
    1 & 0 & 0 & \dots & 0 \\
    \lambda & 1 & 0 & \dots & 0 \\
    \lambda^2 & \lambda & 1 & \dots & 0 \\
    \vdots & \vdots & \vdots & \ddots & 0 \\
    \lambda^{t} & \lambda^{t-1} & \lambda^{t-2} & \dots & 1
\end{bmatrix}.
$$ 
此時無法直接利用結合律。需要寫成 recurive form 才能利用結合律 (先 kv 後 q) 。每個 output row, $\operatorname{o}_t$, 代表一個時間 $t$.  此處省略 Norm.  **但是此處多加入一個 $\lambda$ 作爲 forgetting factor.**  
$$
\mathbf{o}_t= \mathbf{q}_t \sum _{s \leq t}{\lambda}^{t-s} \mathbf{k}_s^{\top} \mathbf{v}_s .
$$
推導過程如下，利用 recursive form 從 0 開始。 the above equation can be rewritten as,  此處 $\mathbf{kv}$ 是一個變數！
$$
\begin{aligned}
\mathbf{k v}_0 & =0 \in \mathbb{R}^{d \times d} \\
\mathbf{k v}_t & =\lambda \, \mathbf{k v}_{t-1}+\mathbf{k}_t^{\top} \mathbf{v}_t \\
\mathbf{o}_t & =\mathbf{q}_t\left(\mathbf{k v}_t\right)
\end{aligned}
$$

where

$$
\mathbf{k} \mathbf{v}_t=\sum_{s \leq t} \lambda^{t-s} \mathbf{k}_s^{\top} \mathbf{v}_s
$$



## Block Diagram 比較


![[Pasted image 20241220210142.png]]








## Non-causal Bi-Directional Attention Kernel
![[Pasted image 20241111233110.png]]

4. Fast-Attention (FA),
5. **P**ositive Random Feature (**+/P**RF),
6. Orthogonal Random features (ORF).

### **Comparison Table: Attention Only**

| **Aspect**                 | **Normal Transformer** | **Linear Transformer** |
| -------------------------- | ---------------------- | ---------------------- |
| **Training Computation**   | $O(n^2 \cdot d)$       | $O(n \cdot d^2)$       |
| **Training Storage**       | $O(n^2 + n \cdot d)$   | $O(n \cdot d)$         |
| **Prefill Computation**    | $O(n^2 \cdot d)$       | $O(n \cdot d^2)$       |
| **Prefill Storage**        | $O(n^2 + n \cdot d)$   | $O(n \cdot d)$         |
| **Generation Computation** | $O(n^2 \cdot d)$       | $O(n \cdot d^2)$       |
| **Generation Storage**     | $O(n \cdot d)$         | $O(n \cdot d)$         |

Linear Attention or Linear Transformer 看起來很美好，有兩個問題：
7. Error：目前的 linear attention 大多是用 random feature 產生，和原來的 transformer 有一定的誤差。而且收斂速度非常慢。所以尚未普及。
8. Causal attention.   以上都是 non-causal attention.  如果是 causal:  對於 normal transformer 基本就是在 softmax(Q K' +M)   加上一個 mask.  基本沒有任何的計算或 storage overhead.  事實上還減少 overhead.  但是對於 linear attention 就不是如此！
	1. Causal linear attention:  對於 generation phase, 沒有任何問題，和 non causal linear attention 一樣。基本就是 RNN.
	2. Causal linear attention: 最大問題是在 training phase.  因爲會變成同時要產生多個 outputs for training.  如果是用類似 RNN iterative 方法，非常沒有效率。這是有名的 cumsum 問題！
	3. Causal linear attention: 對於 pre-fill phase,  沒有問題！！！因爲 pre-fill phase 只在乎最後的 token for the upcoming generation!  所以可以平行計算。


## Causal Linear Attention

以下是 performer 的 training 部分的説明。注意的是 uni-directional 也就是 causal 的部分。
**我是看不懂以下 unidirectional (causal) 部分的 G matrix。直接看 code 還比較清楚。就是 iterative 的算法**
![[Pasted image 20241117222056.png]]



數學的比較。


### 白話文 Cumsum

### Implementing Causality with Cumulative Sums

By using cumulative sums, we can efficiently compute the causal attention outputs:

- **Cumulative Sum of Keys**:
  - $S_t = \sum_{i=1}^{t} k_i = S_{t-1} + k_t$
- **Cumulative Sum of Context**:
  - $Z_t = \sum_{i=1}^{t} k_i v_i^\top = Z_{t-1} + k_t v_t^\top$
- **Normalization Factor**:
  - $D_t = q_t^\top S_t + \epsilon$
- **Attention Output**:
  - $\text{out}_t = \frac{1}{D_t} q_t^\top Z_t$

This ensures that at each position $t$, the output depends only on keys and values from positions $\leq t$

有兩種模型：
9. Recursive 類似 RNN 方法。好處是 inference generation 節省算力和 storage
10. parallel 方法:  用於 training 和 prefill.  不過兩者還是不同。 
	- Prefill 只需要 prompt 最後的 $out_t$.
	- Training 則需要所有的 $out_1, out_2, ..., out_t$

不止考慮算力，還要考慮 memory access.   計算 $out_1, out_2, .., out_t$ 需要大量 memory access.  而 non-causal 只需要一次。  

問題不在於 inferencing, 不論是 pre-fill 或是 generation.  因爲 $S_t$ 和 $Z_t$ 都可以一次運算，或是平行運算。注意
- Prefill 只需要 $out_t$: 也就是 $Z_t$, $q_t$, 和 $S_t$.  當然要得到 $Z_t$ 還是需要 $k_1, k_2, ..., k_t$ 和 $v_1, v_2, ..., v_t$.
- Generation 更簡單，只需要之前的  $Z_t$, $q_t$, 和 $S_t$. 
- 問題是 training,  需要所有的 $out_1, out_2, ..., out_t$ , 也就是所有的 $Z_1, Z_2, ..., Z_t$  和 $S_1, S_2, ..., S_t$
	- $S_1, S_2, ..., S_t$  沒有問題，就是 $k_1, k_2, .., k_t \in R^{1\times d}$  乘以 lower triangular matrix with all-1 elements.
	- $Z_1, Z_2, ..., Z_t$ 應該要特別主要,  $k_1 v_1, k_2 v_2, ..., k_t v_t \in R^{d\times d}$  也是可以乘 乘以 lower triangular matrix with all-1 elements?



## 如何加速 Causal Linear Attention in Training (and Inferencing)?

Linear attention 的好處是在 n >> d 的時候。
因此一個想法就是, n <= d 作為一個 chunk,  仍然採用一般 transformer (quadratic attention) 方法。沒有 training causal 的問題。
但是在 n > d 時候，分成一個個 chunk, 採用 linear attention 方法！因為是以 chunk 為單位。不會需要每個 q.  

答案是 mixed linear attention and quadratic attention using chunk processing.

11. Pure transformer mode: 問題是 long context 的 training 和 inferencing 時 computation 和 memory 都和長度平方正比的問題。
12. Pure sequential mode: 基本和 RNN 一樣。問題是對於 training 太慢。對於 inferencing 的 pre-fill (prompt mode) stage 也太慢。但是對於 inferencing 的 generation mode 非常好。因為 NO KV cache 固定 memory and token time。
13. (Non-materialization) Chunk sequential mode (下圖 a):  分成 chunks

$$
\begin{aligned}
& \mathbf{S}_{[i+1]}=\mathbf{S}_{[i]}+\underbrace{\sum_{j=i C+1}^{(i+1) C} \boldsymbol{k}_j^{\top} \boldsymbol{v}_j}_{\mathbf{K}_{[i]}^{\top} \mathbf{V}_{[i]}} \in \mathbb{R}^{d \times d} . \\
& \mathbf{O}_{[i+1]}=\underbrace{\mathbf{Q}_{[i+1]} \mathbf{S}_{[i]}}_{\text {inter-chunk: } \mathbf{O}_{[i+1]}^{\text {inexe }}}+\underbrace{\left(\left(\mathbf{Q}_{[i+1]} \mathbf{K}_{[i+1]}^{\top}\right) \odot \mathbf{M}\right) \mathbf{V}_{[i+1]}}_{\text {intra-chunk: } \mathbf{O}_{[i+1]}^{\text {intra }}},\\
&\text { where } \mathbf{O}_{[i+1]} \in \mathbb{R}^{C \times d} 
\end{aligned}
$$

 - 在 inter-check 用 linear attention 計算 $S_n$.  
 - 在 intra-chunk 使用 pure transformer mode $O_{intra}$ 加上 initial $O_{inter}$, 因為一次可以產生所有的 outputs (利用上式).  也只需要存一個 chunk 的 (Q)KV cache (cxdx3) 和一個 S (dxd) 
 - 好處是節省 memory.  壞處是雖然速度比 pure sequential mode，但因為整個過程仍然是 sequential, 並非是最快的方法。

![[Pasted image 20241120175328.png]]

- 這個方法讓我想起神經系統的 myelination **髓鞘**。信號在 Ranvier node (蘭氏結) 之間加速跳躍。類似 chunk sequential mode. 
- Chunk sequential mode 適合 inferencing 的 prompt (prefill) mode.  
![[Pasted image 20241120215549.png]]

14. (Materialization) Mixed sequential (上圖 b) + chunk-wise parallel mode (上圖 c):  基本結構和 3 完全一樣。差異是順序。
- 先計算所有 $S_n$  in sequence.  注意這是 in token chunk (c=128 or higher), 所以比 2 的 sequence in token 要快的多。
- 有了所有的 $S_n$, 接下來就是平行計算所有的 outputs.  因為這些 chunks 之間都沒有直接關係 (都是經過 $S_n$),  可以放在 batch dimension (類似 FlashAttention).   


![[Pasted image 20241120181249.png]]


## Causal Attention Training and Inferencing

|                        | Mode1 | Mode2 | Mode3                     | Mode4 |
| ---------------------- | ----- | ----- | ------------------------- | ----- |
| Training-Foward        |       |       |                           | V     |
| Training-Backward      |       |       |                           |       |
| Inferencing-Prompt     |       |       | V.....(No, 只要半套，就是 $S_n$) |       |
| Inferencing-Generation |       | V     |                           |       |

### 注意在 inferencing prompt mode, 其實只要做 mode 3 的半套，就是 $S_n$.  在最後一個 block 計算 output.  因為 prompt mode 只 care 產生 output token 的哪一個 output 即可。不在乎之前的 o.

也可以視為 mode 2.  就是先 compute 所有的 kv sum, 可以平行計算 (例如用 binary tree)，最後 sum.  然後一次得到最後的 output.   因為 prompt (prefill) mode 不在意之前的 output.  只有 training 才在乎。

$$
\begin{aligned}
& \mathbf{S}_{[i+1]}=\mathbf{S}_{[i]}+\underbrace{\sum_{j=i C+1}^{(i+1) C} \boldsymbol{k}_j^{\top} \boldsymbol{v}_j}_{\mathbf{K}_{[i]}^{\top} \mathbf{V}_{[i]}} \in \mathbb{R}^{d \times d} . \\
& \mathbf{O}_{last}=\underbrace{\mathbf{Q}_{last} \mathbf{S}_{last-1}}_{\text {inter-chunk: } \mathbf{O}_{last}^{\text {inexe }}}+\underbrace{\left(\left(\mathbf{Q}_{last} \mathbf{K}_{last}^{\top}\right) \odot \mathbf{M}\right) \mathbf{V}_{last}}_{\text {intra-chunk: } \mathbf{O}_{[i+1]}^{\text {intra }}},
\end{aligned}
$$



## Another Interesting Paper:  引入 decay rate

https://arxiv.org/pdf/2401.04658

![[Pasted image 20241202224214.png]]


### **原始的問題，先是 non-causal 
可以利用 linear attention 的結合律如下：完美的變成 low rank.
$$
\begin{aligned}
& \mathbf{O}=\operatorname{Norm}\left(\left(\mathbf{Q K}^{\top}\right) \mathbf{V}\right), \\
& \mathbf{O}=\operatorname{Norm}\left(\mathbf{Q}\left(\mathbf{K}^{\top} \mathbf{V}\right)\right), \\
\end{aligned}
$$
### 如果是 causal mask，同時假設 M 有遺忘 factor
$$
\begin{aligned}
& \mathbf{O}=\operatorname{Norm}\left[\left(\left(\mathbf{Q K}^{\top}\right) \odot \mathbf{M} \right)\mathbf{V}\right],
\end{aligned}
$$

A lower triangular matrix for a causal mask, $\mathbf{M}$, ensures that a token can only attend to itself and tokens before it, preventing future tokens from influencing the current one. Alternatively, this can be represented in matrix form for a sequence of length $t$:

$$
M =
\begin{bmatrix}
    1 & 0 & 0 & \dots & 0 \\
    \lambda & 1 & 0 & \dots & 0 \\
    \lambda^2 & \lambda & 1 & \dots & 0 \\
    \vdots & \vdots & \vdots & \ddots & 0 \\
    \lambda^{t} & \lambda^{t-1} & \lambda^{t-2} & \dots & 1
\end{bmatrix}.
$$ 
此時無法直接利用結合律。需要寫成 recurive form 才能利用結合律 (先 kv 後 q) 。每個 output row, $\operatorname{o}_t$, 代表一個時間 $t$.  此處省略 Norm.  **但是此處多加入一個 $\lambda$ 作爲 forgetting factor.**  
$$
\mathbf{o}_t= \mathbf{q}_t \sum _{s \leq t}{\lambda}^{t-s} \mathbf{k}_s^{\top} \mathbf{v}_s .
$$
推導過程如下，利用 recursive form 從 0 開始。 the above equation can be rewritten as,  此處 $\mathbf{kv}$ 是一個變數！
$$
\begin{aligned}
\mathbf{k v}_0 & =0 \in \mathbb{R}^{d \times d} \\
\mathbf{k v}_t & =\lambda \, \mathbf{k v}_{t-1}+\mathbf{k}_t^{\top} \mathbf{v}_t \\
\mathbf{o}_t & =\mathbf{q}_t\left(\mathbf{k v}_t\right)
\end{aligned}
$$

where

$$
\mathbf{k} \mathbf{v}_t=\sum_{s \leq t} \lambda^{t-s} \mathbf{k}_s^{\top} \mathbf{v}_s
$$
這裏的問題是 recursive 對於 training 和 prefill 因爲無法平行，速度太慢，是票房毒藥。

因此就有所謂的 chunk method 或是 tiling method.

### **Chunk/Tiling Method

To perform tiling, let us write the equations in block form. Given the total sequence length $n$ and block size $B, \mathbf{X}$ is divided into $T=\frac{n}{B}$ blocks $\left\{\mathbf{X}_1, \mathbf{X}_2, \ldots, \mathbf{X}_T\right\}$ of size $B \times d$ each, where $\mathbf{X} \in\{\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O}\}$.

We first define
$$
\mathbf{K} \mathbf{V}_0=\mathbf{0} \in \mathbb{R}^{d \times d}, \mathbf{K} \mathbf{V}_t=\sum_{s \leq t B} \lambda^{t B-s} \mathbf{k}_s^{\top} \mathbf{v}_s
$$

Given $\mathbf{K} \mathbf{V}_t$ (也就是前面的 $\mathbf{S}_{[t]}$), the output of $(t+1)$-th block, i.e., $t B+r$, with $1 \leq r \leq B$ is
$$
\begin{aligned}
& \mathbf{o}_{t B+r} \\
= & \mathbf{q}_{t B+r} \sum_{s \leq t B+r} \lambda^{t B+r-s} \mathbf{k}_s^{\top} \mathbf{v}_s \\
= & \mathbf{q}_{t B+r}\left(\sum_{s=t B+1}^{t B+r} \lambda^{t B+r-s} \mathbf{k}_s^{\top} \mathbf{v}_s+\lambda^r \sum_{s \leq t B} \lambda^{t B-s} \mathbf{k}_s^{\top} \mathbf{v}_s\right) \\
= & \mathbf{q}_{t B+r} \sum_{s=t B+1}^{t B+r} \lambda^{t B+r-s} \mathbf{k}_s^{\top} \mathbf{v}_s+\lambda^r \mathbf{q}_{t B+r} \mathbf{k v}_{t B} .
\end{aligned}
$$
Rewritten in matrix form, we have
$$
\begin{aligned}
&\begin{aligned}
\mathbf{O}_{t+1}= & \underbrace{\left[\left(\mathbf{Q}_{t+1} \mathbf{K}_{t+1}^{\top}\right) \odot \mathbf{M}\right] \mathbf{V}_{t+1}}_{\text {Intra Block }} \\
& +\underbrace{\Lambda \mathbf{Q}_{t+1}\left(\mathbf{K V}_t\right)}_{\text {Inter Block }},
\end{aligned}\\
&\text { where }\\
&\begin{aligned}
\mathbf{M}_{s t} & = \begin{cases}\lambda^{s-t} & s \geq t \\
0 & s<t\end{cases} \\
\Lambda & =\operatorname{diag}\left\{1, \ldots, \lambda^{B-1}\right\} .
\end{aligned}
\end{aligned}
$$



## Source

- DiJiang Q&A:  https://github.com/YuchuanTian/DiJiang/issues/6
- Lightning attention:  https://arxiv.org/pdf/2401.04658
- Transformer in linear time, FLASH (not Flash Attention):  https://arxiv.org/pdf/2202.10447
- [[2023-03-26-Transformer_LLM]] , [[2023-02-18-Attn_All_U_Need_Visual]]
- https://www.kexue.fm/archives/7546/comment-page-1: check this article
- FAST algorithm: https://arxiv.org/pdf/2009.14794
- https://teddykoker.com/2020/11/performers/
- https://teddykoker.com/2020/11/performers/
- Linear Attention 打改变 Transformer 大模型结构垄断 : https://www.bilibili.com/video/BV1V7s9etEmQ/?spm_id_from=333.999.0.0&vd_source=a99fc6374f18662fe559d32fdc3a80cd
- Transformer are RNNs: https://arxiv.org/pdf/2006.16236
- TransNormerLLM:  https://arxiv.org/pdf/2307.14995
- Universal Transformer:  https://arxiv.org/pdf/1807.03819
- Transformer Quality in Linear Time: https://arxiv.org/pdf/2202.10447





