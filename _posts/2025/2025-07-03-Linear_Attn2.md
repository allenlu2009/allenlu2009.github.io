---
title: Linear Attention Math Framework
date: 2025-07-03 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2025-0324-Linear_Attn]]"
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

1. **線性注意力簡史**： https://spaces.ac.cn/archives/11033
2. (Very good！) Lee Hung-Yi YouTube: https://www.youtube.com/watch?v=gjsdVi90yQo&ab_channel=Hung-yiLee
3. (ByteDance paper) Understanding Transformer from the Perspective of Associative Memory  https://papers.cool/arxiv/2505.19488
4. 


說到超越Softmax Attention，開頭提到，如今的線性Attention不僅能與Softmax Attention一較高低 ，甚至開始＂反哺＂它。這看似不可思議，但細思之下並不難理解。某種意義上，這些年Softmax Atte ntion一直在退步，從MHA、GQA到MQA都是爲了壓縮KV Cache而做減法。而線性Attention沒有K V Cache問題，所以一直往更好的方向前進。

## Normal Transformer

先定義 Transformer notations：
$$
\begin{aligned}
& \boldsymbol{q}_i, \boldsymbol{k}_i, \boldsymbol{v}_i, \boldsymbol{o}_i \in \mathbb{R}^{d \times 1} \\
\boldsymbol{Q}= & {\left[\boldsymbol{q}_1, \boldsymbol{q}_2, \cdots, \boldsymbol{q}_n\right]^{\top} \in \mathbb{R}^{n \times d} } \\
\boldsymbol{K}= & {\left[\boldsymbol{k}_1, \boldsymbol{k}_2, \cdots, \boldsymbol{k}_n\right]^{\top} \in \mathbb{R}^{n \times d} } \\
\boldsymbol{V}= & {\left[\boldsymbol{v}_1, \boldsymbol{v}_2, \cdots, \boldsymbol{v}_n\right]^{\top} \in \mathbb{R}^{n \times d} } \\
\boldsymbol{O}= & {\left[\boldsymbol{o}_1, \boldsymbol{o}_2, \cdots, \boldsymbol{o}_n\right]^{\top} \in \mathbb{R}^{n \times d} }
\end{aligned}
$$

一個Attention模型，本質上是一個 $\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V} \rightarrow \boldsymbol{O}$ 的映射。本文主要關心 Causal 場景，這意味着 $\boldsymbol{o}_t$ 至多跟 $\boldsymbol{Q}_{[: t]}, \boldsymbol{K}_{[: t]}, \boldsymbol{V}_{[: t]}$ 相關 •

標準的Softmax Attention，通常是指 《Attention is All You Need》所提的Attention機制：

$$
\boldsymbol{O}=\operatorname{softmax}\left(\boldsymbol{Q} \boldsymbol{K}^{\top}+\log \boldsymbol{M}\right) \boldsymbol{V}
$$

這裏省略了縮放因子 $1 / \sqrt{d}$ ，因爲它總可以吸收到 $\boldsymbol{Q}, \boldsymbol{K}$ 裏邊，softmax是對第二個維度進行指數歸一化，而 $M \in \mathbb{R}^{n \times n}$ 是一個下三角陣，稱爲掩碼矩陣，定義爲

$$
M_{i, j}= \begin{cases}1, & i \geq j \\ 0, & i<j\end{cases}
$$

$\log \boldsymbol{M}$ 是指對 $\boldsymbol{M}$ 的分量逐一取 $\log$ ，其中 $\log 0=-\infty$ 。Softmax Attention用分量形式寫出來則是

$$
\boldsymbol{o}_t=\frac{\sum_{j=1}^t \exp \left(\boldsymbol{q}_t^{\top} \boldsymbol{k}_j\right) \boldsymbol{v}_j}{\sum_{j=1}^t \exp \left(\boldsymbol{q}_t^{\top} \boldsymbol{k}_j\right)}
$$

其中分母的作用主要是保持數值穩定性，另外就是如果我們給 $\boldsymbol{O}$ 加上RMSNorm，那麼分母也會自動消去，所以Softmax Attention的核心是分子部分，即

$$
\boldsymbol{O}=\exp \left(\boldsymbol{Q} \boldsymbol{K}^{\top}+\log \boldsymbol{M}\right) \boldsymbol{V}=\left(\exp \left(\boldsymbol{Q} \boldsymbol{K}^{\top}\right) \odot \boldsymbol{M}\right) \boldsymbol{V}
$$


其中 $\odot$ 是Hadamard積， $\exp$ 是逐分量取指數。不難看出，分母其實就是將 $\boldsymbol{V}$ 換成一個 $n \times 1$ 的全 1 矩陣，如果有需要，我們再補上即可。Softmax Attention的標準實現需要把 $n \times n$ 的矩陣 $\exp \left(\boldsymbol{Q} \boldsymbol{K}^{\top}\right)$算出來，所以空間和時間複雜度都正比於 $n^2$ 。**Flash Attention的出現降低了空間需求，但平方的時間複雜度依然無法避免。**


## Linear Attention Evolution 和反哺

說到超越Softmax Attention，開頭提到，如今的線性Attention不僅能與Softmax Attention一較高低 ，甚至開始＂反哺＂它。這看似不可思議，但細思之下並不難理解。某種意義上，這些年Softmax Atte ntion一直在退步，從MHA、GQA到MQA都是爲了壓縮KV Cache而做減法。而線性Attention沒有K V Cache問題，所以一直往更好的方向前進。

爲了更好看出這一點，我們不妨將前面提到的Attention機制都以矩陣形式寫出來：

|  | 公式 |
| :--- | :--- |
| Softmax Attention | $\left(\exp \left(\boldsymbol{Q} \boldsymbol{K}^{\top}\right) \odot \boldsymbol{M}\right) \boldsymbol{V}$ |
| 最早的線性Attention | $\left(\boldsymbol{Q K}^{\top} \odot \boldsymbol{M}\right) \boldsymbol{V}$ |
| 加入遺忘門後 | $\left(\boldsymbol{Q} \boldsymbol{K}^{\top} \odot \boldsymbol{\Gamma}\right) \boldsymbol{V}$ |
| DeltaNet | $\left(\boldsymbol{Q} \boldsymbol{K}^{\top} \odot \boldsymbol{M}\right)\left(\boldsymbol{I}+\boldsymbol{K} \boldsymbol{K}^{\top} \odot \boldsymbol{M}^{-}\right)^{-1} \boldsymbol{V}$ |
| Gated DeltaNet | $\left(\left(\boldsymbol{Q} \boldsymbol{K}^{\top} \odot \boldsymbol{M}\right)\left(\boldsymbol{I}+\boldsymbol{K} \boldsymbol{K}^{\top} \odot \boldsymbol{M}^{-}\right)^{-1} \odot \boldsymbol{\Gamma}\right) \boldsymbol{V}$ $=\left(\boldsymbol{Q} \boldsymbol{K}^{\top} \odot \boldsymbol{\Gamma}\right)\left(\boldsymbol{I}+\boldsymbol{K} \boldsymbol{K}^{\top} \odot \boldsymbol{\Gamma}^{-}\right)^{-1} \boldsymbol{V}$ |

其中
$$
\Gamma_{i, j}=\left\{\begin{array}{cc}
\prod_{\tau=j+1}^i \gamma_\tau, & i>j \\
1, & i=j \\
0, & i<j
\end{array}\right.
$$

以及 $\boldsymbol{\Gamma}^{-}=\boldsymbol{\Gamma}-\boldsymbol{I}$ 。這樣看來，Softmax Attention的形式還僅停留在最早的線性Attention那會（當然這也證明了它的強大）。

### Stage 1 - Transformer 便宜的替代

首先我們需要一種方法把Softmax Attention轉化爲線性Attention，這個並不難，早在 《Transformer升級之路：5、作爲無限維的線性Attention》我們就總結了三種將Softmax Attention轉化爲無限維線性Attention的方案。

- 暴力法：直接把 softmax 換成 ReLU or other separable functions 
- Kernel 法
- ..

### Linear Attention: Causal vs Non-Causal

**Non-causal:**

$$
(\boldsymbol{Q} \boldsymbol{K}^{T}) \boldsymbol{V} = \boldsymbol{Q} (\boldsymbol{K}^{T} \boldsymbol{V})
$$
複雜度：$\mathcal{O}(n d^2)$


**Causal (recursive form):**

$$
\boldsymbol{o}_t = \boldsymbol{S}_t \boldsymbol{q}_t, \quad \text{where} \quad \boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{T}
$$

**Pros:** $\mathcal{O}(n)$ complexity
**Cons:** Performance not good

### Performance Improvement

1. **加上 Forgetting factor:**

$$
\boldsymbol{S}_t = \gamma \boldsymbol{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{T}
$$

2. **Input-data dependent decay/emphasis:**
   Mamba v1/v2, RNN-style enhancements, etc.

3. **Cumulative sum (cumsum) is key!**
   What about **selective scan**?


### Stage 2 – Testing Time Training (TTT)

從下式出發（視為一個類似 Optimizer 的過程)，看起來是否很熟悉？像是 gradient descent GD.

$$
\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{T}
$$

進而定義 loss function：

$$
\mathcal{L} = -\boldsymbol{v}^{T} (\boldsymbol{S} \boldsymbol{k})
$$

為了提升穩定性，加入 regularization term。

---

最早期的 loss function 僅為：
$$
\mathcal{L} = -\boldsymbol{v}^{T} (\boldsymbol{S} \boldsymbol{k})
$$
對應的 gradient descent (假設 learning rate $\eta_t=1$):
$$
\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{T}
$$
但這個 loss 沒有下界，是不穩定的。

簡單改善：加入 regularization，例如是 $\|\boldsymbol{S}\|^2$ 的形式，

$$
\mathcal{L} = -\boldsymbol{v}^{T} (\boldsymbol{S} \boldsymbol{k}) + \frac{1-\gamma}{2}\|\boldsymbol{S}\|^2
$$
對應的 gradient descent (假設 learning rate $\eta_t=1$):
$$
\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{T} - (1-\gamma) \boldsymbol{S}_{t-1} = \gamma \boldsymbol{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{T}
$$
$\gamma$ 剛好就是 forgetting factor.  越接近 1 代表遺忘很慢。反之則是以最新的資訊為主。


### Comparison of Update Rules

但若是隻有 $\|\boldsymbol{S}\|^2$ 的形式，並不鼓勵 $\boldsymbol{S} \boldsymbol{k} = \boldsymbol{v}$ 的收斂。所以改成 $\|\boldsymbol{S} \boldsymbol{k} - \boldsymbol{v}\|^2$

$$
\mathcal{L} = \dfrac{1}{2}\|\boldsymbol{S} \boldsymbol{k} - \boldsymbol{v}\|^2
$$

對 loss function 取 gradient on $S$ 並假設 learning rate $\eta_t$
$$\begin{align}
\boldsymbol{S}_t &= \boldsymbol{S}_{t-1} - \eta_{t} \nabla_{S_{t-1}}  \mathcal{L}\\
&=  \boldsymbol{S}_{t-1} - \eta_t ( \boldsymbol{S}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t) \boldsymbol{k}_t^{T}
\end{align}$$

### Why $\boldsymbol{S} \boldsymbol{k} = \boldsymbol{v}$ Matters?

這就要提到 associated memory 的觀念，這是 machine learning 的另一支。Associated memory 是把過去的資訊 (key, value) 壓縮在固定的 memory.   如何取出資訊不是用 sequential access 或是 random access, 而是輸入 key 會得到對應的 value.  

![[Pasted image 20250705000155.png]]
所以 make sense 讓 S k = v.  最後 S q = v, Yeah!


1. **Retrieval Accuracy**:
   - $\boldsymbol{S}$ is a **compressed (associated) memory** of historical $(\boldsymbol{k}_i, \boldsymbol{v}_i)$ pairs
   - When query $\boldsymbol{q}$ arrives, attention output is $\boldsymbol{o} = \boldsymbol{S}\boldsymbol{q}$
   - $\boldsymbol{S}\boldsymbol{k} \approx \boldsymbol{v}$ ensures the memory **accurately reconstructs values** when probed with their original keys
   - *Analogy*: Like testing if a dictionary returns correct definitions when queried with known words

2. **Stability via Bounded Optimization**:
   - The loss $\mathcal{L} = \frac{1}{2}\|\boldsymbol{S}\boldsymbol{k} - \boldsymbol{v}\|^2$ has:
     - Clear minimum at 0 (when $\boldsymbol{S}\boldsymbol{k} = \boldsymbol{v}$)
     - Quadratic growth away from minimum → guarantees convergence
   - Contrast with earlier loss $-\boldsymbol{v}^T(\boldsymbol{S}\boldsymbol{k})$ which has **no lower bound**

3. **Error-Correcting Feedback**:
   The update contains a self-correcting term:
   $$
   \Delta\boldsymbol{S} = -\eta_t \underbrace{(\boldsymbol{S}_{t-1}\boldsymbol{k}_t - \boldsymbol{v}_t)}_{\text{prediction error}} \boldsymbol{k}_t^{T}
   $$
   - When $\boldsymbol{S}_{t-1}\boldsymbol{k}_t$ overestimates $\boldsymbol{v}_t$, update **reduces** $\boldsymbol{S}$
   - When it underestimates, update **increases** $\boldsymbol{S}$

可以更複雜一點：

$$
\mathcal{L} = \dfrac{1}{2}\|\boldsymbol{S} \boldsymbol{k} - \boldsymbol{v}\|^2 + \frac{1-\gamma}{2}\|\boldsymbol{S}\|^2
$$
對應的 gradient descent (假設 learning rate $\eta_t=1$):
$$
\boldsymbol{S}_t = \boldsymbol{S}_{t-1} -\eta_t \underbrace{(\boldsymbol{S}_{t-1}\boldsymbol{k}_t - \boldsymbol{v}_t)}_{\text{prediction error}} \boldsymbol{k}_t^{T} - (1-\gamma) \boldsymbol{S}_{t-1} = \gamma \boldsymbol{S}_{t-1} - \boldsymbol{S}_{t-1}\boldsymbol{k}_t \boldsymbol{k}_t^{T}+ \boldsymbol{v}_t \boldsymbol{k}_t^{T}
$$
$\gamma$ 是 forgetting factor.  $\eta_t = 1$ 不影響 generality, 因為可以把 $\eta_t = \sqrt{\eta_t}\sqrt{\eta_t}$ 可以 factor-in $v_t$ and $k_t$。

### (Good) Comparison of Update Rules

| **Method**                  | **Update Rule**                                  | **Stability** | **Retrieval Accuracy** |
|-----------------------------|--------------------------------------------------|---------------|------------------------|
| Original (unstable)         | $\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{v}_t\boldsymbol{k}_t^T$ | ❌            | ❌                    |
| Forgetting Factor ($\gamma$) | $\boldsymbol{S}_t = \gamma\boldsymbol{S}_{t-1} + \boldsymbol{v}_t\boldsymbol{k}_t^T$ | ✅            | ❌                    |
| **Prediction Error**        | $\boldsymbol{S}_t = \boldsymbol{S}_{t-1} - \eta_t(\boldsymbol{S}_{t-1}\boldsymbol{k}_t - \boldsymbol{v}_t)\boldsymbol{k}_t^T$ | ✅            | ✅                    |


### DeltaNet: 使用 Delta Rule 進行遞推更新

原始 linear attention 公式：

$$
\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{T}
$$

更新為 Delta Rule 形式：(再假設 $\gamma=1$)

$$
\boldsymbol{S}_t = \boldsymbol{S}_{t-1} - (\boldsymbol{S}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t)\boldsymbol{k}_t^{T} = \boldsymbol{S}_{t-1} (\boldsymbol{I} - \boldsymbol{k}_t \boldsymbol{k}_t^{T}) + \boldsymbol{v}_t \boldsymbol{k}_t^{T}
$$

這種「先減後加」的方式就是所謂的 **Delta Rule**。

 - complexity O(n^2)?
+ Gated input : Gated DeltaNet

![[Pasted image 20250703120815.png]]

![[Pasted image 20250703181815.png]]

### 不同的 loss function，有點類似 Lagrangian!


To facilitate discussions on different memory update methods, we first define a general recurrent form of associative memory as follows: (也稱爲 state equation)

$$
\boldsymbol{S}_t=\boldsymbol{A}_t \boldsymbol{S}_{t-1} \boldsymbol{B}_t+\boldsymbol{C}_t,
$$

where $\boldsymbol{A}_t, \boldsymbol{B}_t$, and $\boldsymbol{C}_t$ are parameter matrices. For example, Eq. 32 is a special case of Eq. 35 when $\boldsymbol{A}_t=\boldsymbol{I}$, $\boldsymbol{B}_t=\boldsymbol{I}$, and $\boldsymbol{C}_t=\boldsymbol{v}_t \boldsymbol{k}_t^{\top}$. What optimization objective corresponds to Eq. 35? Referring to Appendix B.1, we can construct the associated objective function as:

$$
\mathcal{L}_t\left(\boldsymbol{S}_{t-1}\right)=\frac{1}{2} \operatorname{tr}\left(\boldsymbol{S}_{t-1}^{\top} \boldsymbol{S}_{t-1}\right)-\frac{1}{2} \operatorname{tr}\left(\boldsymbol{S}_{t-1}^{\top} \boldsymbol{A}_t \boldsymbol{S}_{t-1} \boldsymbol{B}_t\right)-\operatorname{tr}\left(\boldsymbol{C}_t^{\top} \boldsymbol{S}_{t-1}\right),
$$
Loss function 和 recursive 的關係如下，推導見 Appendix F.

The recurrence equation is equivalent to **gradient descent** with **step size 1**:
$$
\boldsymbol{S}_t = \boldsymbol{S}_{t-1} - \nabla_{\boldsymbol{S}_{t-1}} \mathcal{L}_t
$$


**Table 2**: Different forms of memory update. Different associative memory models, their corresponding parameter matrices ($\boldsymbol{A}_t$, $\boldsymbol{B}_t$, and $\boldsymbol{C}_t$) in recurrent form (Eq. 35), and optimization objectives $\mathcal{L}_t\left(\boldsymbol{S}_{t-1}\right)$. Detailed derivations can be found in Appendix B.2.

| Model                           | $\boldsymbol{A}_{\boldsymbol{t}}$                                      | $\boldsymbol{B}_t$                                                                                   | $\boldsymbol{C}_t$                                                                                                                                                                                      | $\mathcal{L}_t\left(\boldsymbol{S}_{t-1}\right)$                                                                                                                                                                                                                         |
| :------------------------------ | :--------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linear Attention                | $\boldsymbol{I}$                                                       | $\boldsymbol{I}$                                                                                     | $\boldsymbol{v}_{\boldsymbol{t}} \boldsymbol{k}_{\boldsymbol{t}}^{\top}$                                                                                                                                | $-\left\langle \boldsymbol{S}_{t-1} \boldsymbol{k}_t, \boldsymbol{v}_t\right\rangle$                                                                                                                                                                                     |
| Gated Linear Attention          | $\operatorname{diag}\left(\boldsymbol{\lambda}_t\right)$               | $\boldsymbol{I}$                                                                                     | $\boldsymbol{v}_{\boldsymbol{t}} \boldsymbol{k}_{\boldsymbol{t}}^{\top}$                                                                                                                                | $\begin{aligned} &-\left\langle \boldsymbol{S}_{t-1} \boldsymbol{k}_t, \boldsymbol{v}_t\right\rangle \\ &+\frac{1}{2}\left\|\operatorname{diag}\left(\sqrt{1-\boldsymbol{\lambda}_t}\right) \boldsymbol{S}_{t-1}\right\|_{F}^2 \end{aligned}$                            |
| DeltaNet                        | $\boldsymbol{I}$                                                       | $\boldsymbol{I}-\boldsymbol{k}_t \boldsymbol{k}_t^{\top}$                                            | $\boldsymbol{v}_t \boldsymbol{k}_t^{\top}$                                                                                                                                                              | $\frac{1}{2}\left\|\boldsymbol{S}_{t-1} \boldsymbol{k}_t-\boldsymbol{v}_t\right\|^2$                                                                                                                                                                                     |
| DeltaNet + Momentum             | $\boldsymbol{I}$                                                       | $\boldsymbol{I}-\boldsymbol{k}_{\boldsymbol{t}} \boldsymbol{k}_{\boldsymbol{t}}^{\boldsymbol{\top}}$ | $\begin{aligned} &\eta_t \boldsymbol{C}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{\top} \\ &= \sum_{i=1}^t \left(\prod_{j=i+1}^t \eta_j\right) \boldsymbol{v}_i \boldsymbol{k}_i^{\top} \end{aligned}$ | $\begin{aligned} &\frac{1}{2}\left\|\boldsymbol{S}_{t-1} \boldsymbol{k}_t\right\|^2 \\ &-\sum_{i=1}^t \left(\prod_{j=i+1}^t \eta_j\right) \boldsymbol{v}_i^{\top} \boldsymbol{S}_{t-1} \boldsymbol{k}_i \end{aligned}$                                                   |
| Softmax Attention w/o Norm      | $\boldsymbol{I}$                                                       | $\boldsymbol{I}$                                                                                     | $\boldsymbol{v}_t \phi\left(\boldsymbol{k}_t\right)^{\top}$                                                                                                                                             | $-\langle\boldsymbol{S}_{\boldsymbol{t}-\mathbf{1}}\phi\left(\boldsymbol{k}_{\boldsymbol{t}}\right), \boldsymbol{v}_{\boldsymbol{t}}\rangle$                                                                                                                             |
| Softmax Attention w/ Norm* [43] | $\frac{t-1}{t} \boldsymbol{I}$                                         | $\boldsymbol{I}$                                                                                     | $\frac{1}{t} \boldsymbol{v}_t \phi\left(\boldsymbol{k}_t\right)^{\top}$                                                                                                                                 | $\begin{aligned} &-\frac{1}{t}\left\langle\boldsymbol{S}_{t-1} \phi\left(\boldsymbol{k}_t\right), \boldsymbol{v}_t\right\rangle \\ &+\frac{1}{2t}\left\|\boldsymbol{S}_{t-1}\right\|_F^2 \end{aligned}$                                                                  |
| Gated Softmax Attention* [21]   | $\frac{t-1}{t} \operatorname{diag}\left(\boldsymbol{\lambda}_t\right)$ | $\boldsymbol{I}$                                                                                     | $\frac{1}{t} \boldsymbol{v}_t \phi\left(\boldsymbol{k}_t\right)^{\top}$                                                                                                                                 | $\begin{aligned} &-\frac{1}{t}\left\langle \boldsymbol{S}_{t-1} \phi\left(\boldsymbol{k}_t\right), \boldsymbol{v}_t\right\rangle \\ &+\frac{1}{2t}\left\|\operatorname{diag}\left(\sqrt{1-\boldsymbol{\lambda}_t}\right) \boldsymbol{S}_{t-1}\right\|_F^2 \end{aligned}$ |

## ＂反哺＂怎麼實現呢？

簡單說，反哺**是把 element wise 變成 matrix operation again!  再來對比原始的 transformer attention** 看是否可以改善 quality。 
另一個關鍵是 kernel mapping
![[Pasted image 20250706165319.png]]


**如何解釋 differential attention? Appendix G**


|                   | 公式                                                                                                                                                                                                                                                                                                                                                                                                                            |
| :---------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Softmax Attention | $\left(\exp \left(\boldsymbol{Q} \boldsymbol{K}^{\top}\right) \odot \boldsymbol{M}\right) \boldsymbol{V}$                                                                                                                                                                                                                                                                                                                     |
| 最早的線性Attention    | $\left(\boldsymbol{Q K}^{\top} \odot \boldsymbol{M}\right) \boldsymbol{V}$                                                                                                                                                                                                                                                                                                                                                    |
| 加入遺忘門後            | $\left(\boldsymbol{Q} \boldsymbol{K}^{\top} \odot \boldsymbol{\Gamma}\right) \boldsymbol{V}$                                                                                                                                                                                                                                                                                                                                  |
| DeltaNet          | $\left(\boldsymbol{Q} \boldsymbol{K}^{\top} \odot \boldsymbol{M}\right)\left(\boldsymbol{I}+\boldsymbol{K} \boldsymbol{K}^{\top} \odot \boldsymbol{M}^{-}\right)^{-1} \boldsymbol{V}$                                                                                                                                                                                                                                         |
| Gated DeltaNet    | $\left(\left(\boldsymbol{Q} \boldsymbol{K}^{\top} \odot \boldsymbol{M}\right)\left(\boldsymbol{I}+\boldsymbol{K} \boldsymbol{K}^{\top} \odot \boldsymbol{M}^{-}\right)^{-1} \odot \boldsymbol{\Gamma}\right) \boldsymbol{V}$ $=\left(\boldsymbol{Q} \boldsymbol{K}^{\top} \odot \boldsymbol{\Gamma}\right)\left(\boldsymbol{I}+\boldsymbol{K} \boldsymbol{K}^{\top} \odot \boldsymbol{\Gamma}^{-}\right)^{-1} \boldsymbol{V}$ |

其中
$$
\Gamma_{i, j}=\left\{\begin{array}{cc}
\prod_{\tau=j+1}^i \gamma_\tau, & i>j \\
1, & i=j \\
0, & i<j
\end{array}\right.
$$

以及 $\boldsymbol{\Gamma}^{-}=\boldsymbol{\Gamma}-\boldsymbol{I}$ 。這樣看來，Softmax Attention的形式還僅停留在最早的線性Attention那會（當然這也證明了它的強大）。

一直到遺忘門對應的數學都很直覺，也説不上什麼反哺。
主要是從 DeltaNet 開始：
$$
\mathcal{L} = \dfrac{1}{2}\|\boldsymbol{S} \boldsymbol{k} - \boldsymbol{v}\|^2
$$
$$\begin{align}
\boldsymbol{S}_t &= \boldsymbol{S}_{t-1} - \eta_{t} \nabla_{S_{t-1}}  \mathcal{L}\\
&=  \boldsymbol{S}_{t-1} - \eta_t ( \boldsymbol{S}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t) \boldsymbol{k}_t^{T}
\end{align}$$
**是否有對應的 matrix formula?**  不是隻爲了數學簡潔，matrix formula 也可以平行加速運算，特別是針對 GPU 和 NPU.
要完成這一目標，我們先將DeltaNet寫成

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}+\left(\boldsymbol{v}_t-\boldsymbol{S}_{t-1} \boldsymbol{k}_t\right) \boldsymbol{k}_t^{\top}
$$

我們把 prediction error 定義為:  $\boldsymbol{u}_t=\boldsymbol{v}_t-\boldsymbol{S}_{t-1} \boldsymbol{k}_t$ ，那麼 $\boldsymbol{S}_t=\boldsymbol{S}_{t-1}+\boldsymbol{u}_t \boldsymbol{k}_t^{\top}$ ，也就是說它只是在最早的線性Attention基礎上把 $\boldsymbol{V}$換成了 $\boldsymbol{U}=\left[\boldsymbol{u}_1, \boldsymbol{u}_2, \cdots, \boldsymbol{u}_n\right]^{\top}$ ，將它迭代 $t-1$ 次，我們有

$$
\boldsymbol{S}_{t-1}=\sum_{j=1}^{t-1} \boldsymbol{u}_j \boldsymbol{k}_j^{\top} \quad \Rightarrow \quad \boldsymbol{u}_t=\boldsymbol{v}_t-\left(\sum_{j=1}^{t-1} \boldsymbol{u}_j \boldsymbol{k}_j^{\top}\right) \boldsymbol{k}_t=\boldsymbol{v}_t-\sum_{j=1}^{t-1} \boldsymbol{u}_j\left(\boldsymbol{k}_j^{\top} \boldsymbol{k}_t\right)
$$


最後的等式寫成矩陣形式是 $\boldsymbol{U}=\boldsymbol{V}-\left(\boldsymbol{K} \boldsymbol{K}^{\top} \odot \boldsymbol{M}^{-}\right) \boldsymbol{U}$ ，其中 $\boldsymbol{M}^{-}=\boldsymbol{M}-\boldsymbol{I}$ ，這是一個線性方程組，它的解可以直接表示爲

$$
\boldsymbol{U}=(\boldsymbol{I}+\underbrace{\boldsymbol{K} \boldsymbol{K}^{\top} \odot \boldsymbol{M}^{-}}_{\text {記爲 } \boldsymbol{B}})^{-1} \boldsymbol{V}
$$

這裏出現了 $(\boldsymbol{I}+\boldsymbol{B})^{-1}$ ，一個 $n \times n$ 矩陣的逆，標準複雜度是 $\mathcal{O}\left(n^3\right)$ ，比Softmax Attention還高！
不過好在我們不需要顯式的逆而是隻要 $\boldsymbol{U}$，這可以轉化爲解方程組 $\boldsymbol{V}(\boldsymbol{I}+\boldsymbol{B})\boldsymbol{U}=\boldsymbol{V}$，複雜度降到 $\mathcal{O}\left(n^2\right)$。進一步地，利用 $\boldsymbol{I}+\boldsymbol{B}$ 是下三角陣以及 $\boldsymbol{B}$ 的低秩結構，可以將複雜度降到線性，寫成分塊矩陣乘法後就可以充分利用GPU!

簡單說，DeltaNet 的反哺就是把 $\boldsymbol{V}$ 置換成 $\boldsymbol{U}$，所以 linear attention 的 matrix representation 就是：
$$\left(\boldsymbol{Q} \boldsymbol{K}^{\top} \odot \boldsymbol{M}\right)\left(\boldsymbol{I}+\boldsymbol{K} \boldsymbol{K}^{\top} \odot \boldsymbol{M}^{-}\right)^{-1} \boldsymbol{V}$$
對應的原始 attention 就變成:
$$\left(\exp(\boldsymbol{Q} \boldsymbol{K}^{\top}) \odot \boldsymbol{M}\right)\left(\boldsymbol{I}+\exp(\boldsymbol{K} \boldsymbol{K}^{\top}) \odot \boldsymbol{M}^{-}\right)^{-1} \boldsymbol{V}$$
exp 加上 normalization 可以換成 softmax.
$\left(softmax \left(\boldsymbol{Q} \boldsymbol{K}^{\top}\right) \odot \boldsymbol{M}\right) \boldsymbol{V}$



對於 differential attention (Appendix G)

$\left(\text{softmax} \left(\boldsymbol{Q}_1 \boldsymbol{K}_1^{\top}\right) \odot \boldsymbol{M}- \lambda \text{ softmax} \left(\boldsymbol{Q}_2 \boldsymbol{K}_2^{\top}\right) \odot \boldsymbol{M}\right)  \boldsymbol{V}$

Any corresponding recursive form?  $\lambda$ is a training parameter and close to 1 to cancel the common noise and amplify the differential signal.







## Appendix A:  Loss function

### Corrected Loss Function and Update Rule

The corrected loss function is:
$$
\mathcal{L} = \dfrac{1}{2}\|\boldsymbol{S} \boldsymbol{k} - \boldsymbol{v}\|^2
$$

With gradient descent update (learning rate $\eta_t$):
$$
\boldsymbol{S}_t = \boldsymbol{S}_{t-1} - \eta_t (\boldsymbol{S}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t) \boldsymbol{k}_t^{T}
$$

### Why $\boldsymbol{S} \boldsymbol{k} = \boldsymbol{v}$ Matters: Core Reasons

1. **Retrieval Accuracy**:
   - $\boldsymbol{S}$ is a **compressed memory** of historical $(\boldsymbol{k}_i, \boldsymbol{v}_i)$ pairs
   - When query $\boldsymbol{q}$ arrives, attention output is $\boldsymbol{o} = \boldsymbol{S}\boldsymbol{q}$
   - $\boldsymbol{S}\boldsymbol{k} \approx \boldsymbol{v}$ ensures the memory **accurately reconstructs values** when probed with their original keys
   - *Analogy*: Like testing if a dictionary returns correct definitions when queried with known words

2. **Stability via Bounded Optimization**:
   - The loss $\mathcal{L} = \frac{1}{2}\|\boldsymbol{S}\boldsymbol{k} - \boldsymbol{v}\|^2$ has:
     - Clear minimum at 0 (when $\boldsymbol{S}\boldsymbol{k} = \boldsymbol{v}$)
     - Quadratic growth away from minimum → guarantees convergence
   - Contrast with earlier loss $-\boldsymbol{v}^T(\boldsymbol{S}\boldsymbol{k})$ which has **no lower bound**

3. **Error-Correcting Feedback**:
   The update contains a self-correcting term:
   $$
   \Delta\boldsymbol{S} = -\eta_t \underbrace{(\boldsymbol{S}_{t-1}\boldsymbol{k}_t - \boldsymbol{v}_t)}_{\text{prediction error}} \boldsymbol{k}_t^{T}
   $$
   - When $\boldsymbol{S}_{t-1}\boldsymbol{k}_t$ overestimates $\boldsymbol{v}_t$, update **reduces** $\boldsymbol{S}$
   - When it underestimates, update **increases** $\boldsymbol{S}$

### Comparison of Update Rules

| **Method**                  | **Update Rule**                                  | **Stability** | **Retrieval Accuracy** |
|-----------------------------|--------------------------------------------------|---------------|------------------------|
| Original (unstable)         | $\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{v}_t\boldsymbol{k}_t^T$ | ❌            | ❌                    |
| Forgetting Factor ($\gamma$) | $\boldsymbol{S}_t = \gamma\boldsymbol{S}_{t-1} + \boldsymbol{v}_t\boldsymbol{k}_t^T$ | ✅            | ❌                    |
| **Prediction Error**        | $\boldsymbol{S}_t = \boldsymbol{S}_{t-1} - \eta_t(\boldsymbol{S}_{t-1}\boldsymbol{k}_t - \boldsymbol{v}_t)\boldsymbol{k}_t^T$ | ✅            | ✅                    |

### Practical Interpretation
The prediction-error update implements an **online least-squares solver**:
1. $\boldsymbol{S}$ acts as a linear predictor: $\hat{\boldsymbol{v}} = \boldsymbol{S} \boldsymbol{k}$
2. Update follows stochastic gradient descent for:
   $$
   \min_{\boldsymbol{S}} \sum_i \|\boldsymbol{S} \boldsymbol{k}_i - \boldsymbol{v}_i\|^2
   $$
3. Resembles **recursive least squares (RLS)** but with simplified rank-1 update

### Why This Matters for Attention
In causal attention:
- Output at step $t$ is $\boldsymbol{o}_t = \boldsymbol{S}_t \boldsymbol{q}_t$
- $\boldsymbol{S}_t\boldsymbol{k}_j \approx \boldsymbol{v}_j$ ensures:
  - When $\boldsymbol{q}_t$ resembles past $\boldsymbol{k}_j$, output contains accurate $\boldsymbol{v}_j$
  - Prevents "representation drift" in the state matrix
  - Enables exact value retrieval when $\boldsymbol{q} = \boldsymbol{k}_j$

### Key Insight
The condition $\boldsymbol{S}\boldsymbol{k} \approx \boldsymbol{v}$ forces $\boldsymbol{S}$ to be a **functionally correct associative memory** rather than just a mathematically convenient recurrence. This is fundamental for maintaining attention fidelity in linearized models.


## Appendix B: Differential DeltaNet

### DeltaNet 與反哺機制詳解

#### 1. **Delta Rule 的遞推本質**
DeltaNet 的核心是使用 **Delta Rule（梯度下降法）** 進行遞推更新：
$$\boldsymbol{S}_t = \boldsymbol{S}_{t-1} - \eta_t (\boldsymbol{S}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t) \boldsymbol{k}_t^\top$$
其中 $\boldsymbol{u}_t = \boldsymbol{v}_t - \boldsymbol{S}_{t-1} \boldsymbol{k}_t$ 是預測誤差。簡化後得到：
$$\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{u}_t \boldsymbol{k}_t^\top$$
這等價於用誤差 $\boldsymbol{u}_t$ 替代原始線性 Attention 中的 $\boldsymbol{v}_t$。

---

#### 2. **反哺：從遞推到矩陣形式**
反哺的核心思想是**將遞推過程轉化爲全局矩陣運算**，以實現並行化：
1. **展開遞推關係**：
   $$\boldsymbol{u}_t = \boldsymbol{v}_t - \sum_{j=1}^{t-1} (\boldsymbol{k}_j^\top \boldsymbol{k}_t) \boldsymbol{u}_j$$
2. **寫成矩陣方程**：
   $$\boldsymbol{U} = \boldsymbol{V} - (\boldsymbol{K} \boldsymbol{K}^\top \odot \boldsymbol{M}^{-}) \boldsymbol{U}$$
   其中 $\boldsymbol{M}^{-}$ 是嚴格下三角掩碼矩陣（對角爲 0，下三角爲 1）。
3. **解析解**：
   $$\boldsymbol{U} = (\boldsymbol{I} + \boldsymbol{K} \boldsymbol{K}^\top \odot \boldsymbol{M}^{-})^{-1} \boldsymbol{V}$$

---

#### 3. **DeltaNet 的矩陣形式**
將 $\boldsymbol{U}$ 代入線性 Attention 公式：
$$\text{DeltaNet} = (\boldsymbol{Q} \boldsymbol{K}^\top \odot \boldsymbol{M}) \boldsymbol{U} = (\boldsymbol{Q} \boldsymbol{K}^\top \odot \boldsymbol{M}) (\boldsymbol{I} + \boldsymbol{K} \boldsymbol{K}^\top \odot \boldsymbol{M}^{-})^{-1} \boldsymbol{V}$$
- **物理意義**：用預測誤差 $\boldsymbol{U}$ 替代原始值 $\boldsymbol{V}$，實現更精準的梯度修正。
- **複雜度優化**：  
  利用 $\boldsymbol{I} + \boldsymbol{K} \boldsymbol{K}^\top \odot \boldsymbol{M}^{-}$ 的**下三角+低秩**特性：
  - 通過 **Cholesky 分解**或 **Neumann 級數**近似求逆，降至 $O(n d^2)$ 複雜度（$d$ 爲向量維度）。
  - GPU 友好：分塊矩陣乘法實現並行加速。

---

#### 4. **Gated DeltaNet 的推廣**
加入遺忘門 $\gamma_t$ 後，矩陣 $\boldsymbol{\Gamma}$ 定義爲：
$$\Gamma_{i,j} = \begin{cases} \prod_{\tau=j+1}^i \gamma_\tau & i > j \\ 1 & i=j \\ 0 & i < j \end{cases}$$
Gated DeltaNet 的矩陣形式爲：
$$\text{Gated DeltaNet} = (\boldsymbol{Q} \boldsymbol{K}^\top \odot \boldsymbol{\Gamma}) (\boldsymbol{I} + \boldsymbol{K} \boldsymbol{K}^\top \odot \boldsymbol{\Gamma}^{-})^{-1} \boldsymbol{V}$$
其中 $\boldsymbol{\Gamma}^{-} = \boldsymbol{\Gamma} - \boldsymbol{I}$。遺忘門通過 $\gamma_t$ 控制歷史信息衰減強度。

---

#### 5. **Differential Attention 的遞推形式**
原始定義（矩陣形式）：
$$\text{Differential Attention} = \left( \text{softmax}(\boldsymbol{Q}_1 \boldsymbol{K}_1^\top) \odot \boldsymbol{M} + \lambda \text{ softmax}(\boldsymbol{Q}_2 \boldsymbol{K}_2^\top) \odot \boldsymbol{M} \right) \boldsymbol{V}$$
爲設計遞推形式，需分兩步：
1. **獨立維護兩個 DeltaNet 狀態**：
   - $\boldsymbol{S}^{(1)}_t = \gamma^{(1)}_t \boldsymbol{S}^{(1)}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_{1,t}^\top$
   - $\boldsymbol{S}^{(2)}_t = \gamma^{(2)}_t \boldsymbol{S}^{(2)}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_{2,t}^\top$
2. **計算當前步輸出**：
   $$\boldsymbol{o}_t = \underbrace{\boldsymbol{S}^{(1)}_{t-1} \boldsymbol{q}_{1,t}}_{\text{Context from head 1}} + \lambda \underbrace{\boldsymbol{S}^{(2)}_{t-1} \boldsymbol{q}_{2,t}}_{\text{Context from head 2}}$$
   - $\boldsymbol{q}_{1,t}, \boldsymbol{q}_{2,t}$：當前時刻的查詢向量（來自 $\boldsymbol{Q}_1, \boldsymbol{Q}_2$）。
   - 輸出 $\boldsymbol{o}_t$ 是當前時刻的上下文向量。

---

#### 6. **關鍵優勢與物理意義**
- **DeltaNet 的反哺**：  
  將局部遞推誤差 $\boldsymbol{u}_t$ 通過矩陣逆全局傳播，等價於在損失函數 $\mathcal{L} = \frac{1}{2} \|\boldsymbol{S} \boldsymbol{k}_t - \boldsymbol{v}_t\|^2$ 下求最優狀態矩陣 $\boldsymbol{S}$。
- **Differential Attention 的遞推意義**：  
  融合兩個注意力頭的上下文信息：
  - 主頭（$\boldsymbol{S}^{(1)}$）：捕捉核心語義。
  - 輔助頭（$\boldsymbol{S}^{(2)}$）：提供局部修正（$\lambda$ 控制強度）。
  - 類似 **殘差學習機制**，提升模型表達能力。

---

### 總結
| **組件**         | **核心創新**                                                                 | **遞推形式**                                                                 |
|------------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **DeltaNet**     | 用預測誤差 $\boldsymbol{u}_t$ 替代 $\boldsymbol{v}_t$，實現梯度修正               | $\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + (\boldsymbol{v}_t - \boldsymbol{S}_{t-1} \boldsymbol{k}_t) \boldsymbol{k}_t^\top$ |
| **反哺機制**     | 將遞推轉爲矩陣運算 $(\boldsymbol{I} + \boldsymbol{K} \boldsymbol{K}^\top \odot \boldsymbol{M}^{-})^{-1}$ | 通過下三角矩陣求逆實現全局誤差傳播                                              |
| **Gated DeltaNet**| 加入遺忘門 $\gamma_t$ 控制歷史記憶強度                                             | $\boldsymbol{S}_t = \gamma_t \boldsymbol{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top$ |
| **Differential Attention** | 雙注意力頭加權融合                                                          | $\boldsymbol{o}_t = \boldsymbol{S}^{(1)}_{t-1} \boldsymbol{q}_{1,t} + \lambda \boldsymbol{S}^{(2)}_{t-1} \boldsymbol{q}_{2,t}$ |

反哺機制通過**矩陣求逆**將遞推的局部更新轉爲全局優化，是線性 Attention 並行化的關鍵；而 Differential Attention 的遞推形式則通過**雙狀態融合**平衡全局與局部信息，爲動態語義建模提供新範式。

In the context of DeltaNet, the **loss function for differential attention** is derived from its dual-path attention mechanism and DeltaNet's core optimization objective. Here's the formal definition and its components:

---

### **Loss Function Definition**
The global loss function for differential attention is a **weighted sum of two independent DeltaNet losses**, each corresponding to one attention head:

$$
\mathcal{L}_{\text{diff}} = \sum_{t=1}^T \left( \underbrace{\frac{1}{2} \| \boldsymbol{S}^{(1)}_{t-1} \boldsymbol{k}_{1,t} - \boldsymbol{v}_t \|^2}_{\text{Loss for Head 1}} + \lambda \cdot \underbrace{\frac{1}{2} \| \boldsymbol{S}^{(2)}_{t-1} \boldsymbol{k}_{2,t} - \boldsymbol{v}_t \|^2}_{\text{Loss for Head 2}} \right)
$$

Where:
- $\boldsymbol{S}^{(1)}_{t-1}, \boldsymbol{S}^{(2)}_{t-1}$: State matrices for Head 1 and Head 2 at step $t-1$
- $\boldsymbol{k}_{1,t}, \boldsymbol{k}_{2,t}$: Key vectors for each head at step $t$
- $\boldsymbol{v}_t$: Shared value vector at step $t$
- $\lambda$: Weighting hyperparameter (typically $0 < \lambda < 1$)

---

### **Key Components Explained**
1. **Per-Head DeltaNet Loss**:
   Each head independently minimizes the **reconstruction error** between predicted and actual values:
   $$
   \mathcal{L}^{(i)}_t = \frac{1}{2} \| \boldsymbol{S}^{(i)}_{t-1} \boldsymbol{k}_{i,t} - \boldsymbol{v}_t \|^2, \quad i \in \{1,2\}
   $$
   - This follows DeltaNet's original loss design: State matrices $\boldsymbol{S}^{(i)}$ are updated via gradient descent to reconstruct $\boldsymbol{v}_t$ from $\boldsymbol{k}_{i,t}$.

2. **Differential Weighting ($\lambda$)**:
   - $\lambda$ controls the **auxiliary head's influence**:
     - $\lambda \to 0$: Only Head 1 dominates (similar to single-head DeltaNet)
     - $\lambda = 1$: Both heads contribute equally
     - $\lambda > 1$: Auxiliary head (Head 2) corrects dominant errors

3. **Temporal Summation**:
   Loss accumulates over $T$ timesteps, enforcing **sequential error minimization**.

---

### **Physical Intuition**
This loss function implements a **dual-path error correction** mechanism:
1. **Head 1 (Primary)**:
   - Maintains core contextual representation ($\boldsymbol{S}^{(1)} \approx \text{global memory}$)
   - Minimizes major semantic errors
   
2. **Head 2 (Auxiliary)**:
   - Captures local/high-frequency patterns ($\boldsymbol{S}^{(2)} \approx \text{local gradient}$)
   - Corrects residual errors via $\lambda$-scaled loss
   - Acts as a "fine-tuner" for Head 1's output

> **Example**: In language modeling:
> - Head 1 predicts `v_t = "king"` based on global context (e.g., _"The ___ ruled the nation"_)
> - Head 2 corrects to `v_t = "queen"` using local cues (e.g., preceding adjective _"powerful"_)
> - $\lambda$ controls how strongly local cues override global context.

---

### **State Update Rules**
Minimizing $\mathcal{L}_{\text{diff}}$ yields decoupled updates for each head:
$$
\begin{align*}
\boldsymbol{S}^{(1)}_t &= \boldsymbol{S}^{(1)}_{t-1} + \left( \boldsymbol{v}_t - \boldsymbol{S}^{(1)}_{t-1} \boldsymbol{k}_{1,t} \right) \boldsymbol{k}_{1,t}^\top \\
\boldsymbol{S}^{(2)}_t &= \boldsymbol{S}^{(2)}_{t-1} + \lambda \left( \boldsymbol{v}_t - \boldsymbol{S}^{(2)}_{t-1} \boldsymbol{k}_{2,t} \right) \boldsymbol{k}_{2,t}^\top 
\end{align*}
$$
- **Note**: The $\lambda$ scaling **only applies to Head 2's update**, making it adaptively responsive to prediction errors.

---

### **Connection to Output**
The final differential attention output combines both heads:
$$
\boldsymbol{o}_t = \underbrace{\boldsymbol{S}^{(1)}_{t-1} \boldsymbol{q}_{1,t}}_{\text{Head 1 output}} + \lambda \cdot \underbrace{\boldsymbol{S}^{(2)}_{t-1} \boldsymbol{q}_{2,t}}_{\text{Head 2 output}}
$$
- The loss function optimizes $\boldsymbol{S}^{(1)}, \boldsymbol{S}^{(2)}$ to ensure $\boldsymbol{o}_t$ captures both **global coherence** (Head 1) and **local precision** (Head 2).

---

### **Why This Design?**
1. **Error Decomposition**:
   - Separates learning of slow-varying (Head 1) and fast-varying (Head 2) components
   - Matches **spectral properties** of natural sequences (low-freq + high-freq)

2. **Stability**:
   - Primary head ($\lambda$-free) maintains stable base state
   - Auxiliary head ($\lambda$-scaled) makes agile corrections without divergence

3. **Interpretability**:
   - $\lambda$ explicitly controls "how much local context overrides global memory"
   - Adjustable per-task (e.g., $\lambda_{\text{translation}} > \lambda_{\text{summarization}}$)

This loss function enables differential attention to outperform standard DeltaNet in tasks requiring **joint modeling of hierarchical contexts** (e.g., document QA, long-form generation).


## Appendix F: Loss and Gradient Descent

The relationship between the state update equation $S_t = A_t S_{t-1} B_t + C_t$ and the optimization objective $\mathcal{L}_t(S_{t-1})$ is **gradient-based**, where the state transition $S_{t-1} \to S_t$ corresponds to **one step of gradient descent** on $\mathcal{L}_t$. Here's the detailed explanation:

### 1. **Gradient of $\mathcal{L}_t$**
The gradient of the objective $\mathcal{L}_t$ with respect to $S_{t-1}$ is:
$$
\nabla_{S_{t-1}} \mathcal{L}_t = \frac{\partial \mathcal{L}_t}{\partial S_{t-1}} = S_{t-1} - A_t S_{t-1} B_t - C_t
$$
*Derivation*:  
- $\frac{\partial}{\partial S_{t-1}} \left( \frac{1}{2} \operatorname{tr}(S_{t-1}^\top S_{t-1}) \right) = S_{t-1}$  
- $\frac{\partial}{\partial S_{t-1}} \left( -\frac{1}{2} \operatorname{tr}(S_{t-1}^\top A_t S_{t-1} B_t) \right) = -A_t S_{t-1} B_t$ (using symmetry of $A_t, B_t$)  
- $\frac{\partial}{\partial S_{t-1}} \left( -\operatorname{tr}(C_t^\top S_{t-1}) \right) = -C_t$  
- **Sum**: $\nabla_{S_{t-1}} \mathcal{L}_t = S_{t-1} - A_t S_{t-1} B_t - C_t$.

---

### 2. **State Update as Gradient Descent**
The recurrence equation is equivalent to **gradient descent** with **step size 1**:
$$
S_t = S_{t-1} - \nabla_{S_{t-1}} \mathcal{L}_t
$$
Substituting the gradient:
$$
S_t = S_{t-1} - \left( S_{t-1} - A_t S_{t-1} B_t - C_t \right) = \cancel{S_{t-1}} - \cancel{S_{t-1}} + A_t S_{t-1} B_t + C_t
$$
Thus, we recover the original update:
$$
\boxed{S_t = A_t S_{t-1} B_t + C_t}
$$

---

### 3. **Physical Interpretation**
- $\mathcal{L}_t(S_{t-1})$ is a **local objective** that $S_{t-1}$ "optimizes" to become $S_t$.  
- The update minimizes $\mathcal{L}_t$ by moving $S_{t-1}$ in the direction of steepest descent.  
- **Equilibrium**: When $\nabla_{S_{t-1}} \mathcal{L}_t = 0$, we get:
  $$
  S_{t-1} = A_t S_{t-1} B_t + C_t
  $$
  This matches the steady-state condition $S_t = S_{t-1}$ (fixed point).

---

### 4. **Role of Components**
- **Memory Decay**: The $-\frac{1}{2} \operatorname{tr}(S_{t-1}^\top S_{t-1})$ term in $\mathcal{L}_t$ penalizes large state values (regularization).  
- **Linear Transformation**: $-\frac{1}{2} \operatorname{tr}(S_{t-1}^\top A_t S_{t-1} B_t)$ guides state evolution via $A_t, B_t$.  
- **External Input**: $-\operatorname{tr}(C_t^\top S_{t-1})$ aligns $S_{t-1}$ with new data $C_t$.

---

### 5. **Summary of Relationship**
| **Aspect**               | **Mathematical Form**                                  | **Role**                                                                 |
|---------------------------|--------------------------------------------------------|--------------------------------------------------------------------------|
| State Update              | $S_t = A_t S_{t-1} B_t + C_t$                       | Discrete transition rule                                                |
| Optimization Objective    | $\mathcal{L}_t(S_{t-1}) = \frac{1}{2} \operatorname{tr}(\cdots)$ | Local loss function for $S_{t-1}$                                    |
| Gradient                  | $\nabla_{S_{t-1}} \mathcal{L}_t = S_{t-1} - A_t S_{t-1} B_t - C_t$ | Steepest descent direction                                              |
| Update Rule               | $S_t = S_{t-1} - \nabla_{S_{t-1}} \mathcal{L}_t$    | **Exact gradient descent** with step size 1                             |
| Equilibrium Condition     | $\nabla_{S_{t-1}} \mathcal{L}_t = 0$                | Fixed point of the recurrence                                            |

### Key Insight
The recurrence $S_t = A_t S_{t-1} B_t + C_t$ **implicitly optimizes** $\mathcal{L}_t(S_{t-1})$ at each step. This provides a unified optimization perspective for associative memory models, where different choices of $(A_t, B_t, C_t)$ correspond to different objectives (Table 2). The symmetry constraint on $A_t, B_t$ ensures $\mathcal{L}_t$ is well-defined for gradient-based dynamics.



## Appendix G: Differential Linear Attention

To model the associative memory with two states $(\boldsymbol{S}^{(1)}_t, \boldsymbol{S}^{(2)}_t)$ and two keys $(\boldsymbol{k}^{(1)}_t, \boldsymbol{k}^{(2)}_t)$ where information is stored in the difference $\boldsymbol{S}^{(1)}_{t-1} \boldsymbol{k}^{(1)}_t - \boldsymbol{S}^{(2)}_{t-1} \boldsymbol{k}^{(2)}_t$, we define the following components:

### 1. **State Update Equation**
The recurrence relation extends the original form to accommodate two states:
$$
\begin{bmatrix} \boldsymbol{S}^{(1)}_t \\ \boldsymbol{S}^{(2)}_t \end{bmatrix} = \begin{bmatrix} \boldsymbol{A}^{(1)}_t & 0 \\ 0 & \boldsymbol{A}^{(2)}_t \end{bmatrix} \begin{bmatrix} \boldsymbol{S}^{(1)}_{t-1} \\ \boldsymbol{S}^{(2)}_{t-1} \end{bmatrix} \begin{bmatrix} \boldsymbol{B}^{(1)}_t & 0 \\ 0 & \boldsymbol{B}^{(2)}_t \end{bmatrix} + \begin{bmatrix} \boldsymbol{C}^{(1)}_t \\ \boldsymbol{C}^{(2)}_t \end{bmatrix}
$$
**Special Case (Gradient Descent):**  
When $\boldsymbol{A}^{(1)}_t = \boldsymbol{A}^{(2)}_t = I$, $\boldsymbol{B}^{(1)}_t = \boldsymbol{B}^{(2)}_t = I$, and $C_t$ encodes the difference update:
$$
\boxed{
\begin{aligned}
\boldsymbol{S}^{(1)}_t &= \boldsymbol{S}^{(1)}_{t-1} + \left( d_t - (\boldsymbol{S}^{(1)}_{t-1} \boldsymbol{k}^{(1)}_t - \boldsymbol{S}^{(2)}_{t-1} \boldsymbol{k}^{(2)}_t) \right) {\boldsymbol{k}^{(1)}_t}^\top \\
\boldsymbol{S}^{(2)}_t &= \boldsymbol{S}^{(2)}_{t-1} - \left( d_t - (\boldsymbol{S}^{(1)}_{t-1} \boldsymbol{k}^{(1)}_t - \boldsymbol{S}^{(2)}_{t-1} \boldsymbol{k}^{(2)}_t) \right) {\boldsymbol{k}^{(2)}_t}^\top
\end{aligned}
}
$$
- $d_t$: Target value for the difference $\boldsymbol{S}^{(1)}_{t-1} \boldsymbol{k}^{(1)}_t - \boldsymbol{S}^{(2)}_{t-1} \boldsymbol{k}^{(2)}_t$.

---

### 2. **Loss Function**
The objective minimizes the error in the difference while regularizing state changes:
$$
\boxed{
\mathcal{L}_t(\boldsymbol{S}^{(1)}_{t-1}, \boldsymbol{S}^{(2)}_{t-1}) = \frac{1}{2} \left\| (\boldsymbol{S}^{(1)}_{t-1} \boldsymbol{k}^{(1)}_t - \boldsymbol{S}^{(2)}_{t-1} \boldsymbol{k}^{(2)}_t) - d_t \right\|^2 + \frac{\lambda}{2} \left( \|\boldsymbol{S}^{(1)}_{t-1}\|_F^2 + \|\boldsymbol{S}^{(2)}_{t-1}\|_F^2 \right)
}
$$
- **First term**: Ensures $\boldsymbol{S}^{(1)}_{t-1} \boldsymbol{k}^{(1)}_t - \boldsymbol{S}^{(2)}_{t-1} \boldsymbol{k}^{(2)}_t$ matches target $d_t$.  
- **Second term**: Regularization (prevoversus large state values, $\lambda \geq 0$).

---

### 3. **Physics Analogy: Coupled Harmonic Oscillators**
The system resembles coupled oscillators with a "force" driving their difference:
- **Positions**: $q_1 \equiv \boldsymbol{S}^{(1)}_{t-1} \boldsymbol{k}^{(1)}_t$, $q_2 \equiv \boldsymbol{S}^{(2)}_{t-1} \boldsymbol{k}^{(2)}_t$  
- **Potential Energy**: $\frac{1}{2} \left\| (q_1 - q_2) - d_t \right\|^2$ (spring pulling $q_1 - q_2$ toward $d_t$)  
- **Kinetic Energy**: Absent (first-order dynamics, no inertia)  
- **Dynamics**: Overdamped motion toward equilibrium $q_1 - q_2 = d_t$.

---

### 4. **Gradient Derivation**
The gradients of $\mathcal{L}_t$ are:
$$
\begin{aligned}
\nabla_{\boldsymbol{S}^{(1)}_{t-1}} \mathcal{L}_t &= \left( (\boldsymbol{S}^{(1)}_{t-1} \boldsymbol{k}^{(1)}_t - \boldsymbol{S}^{(2)}_{t-1} \boldsymbol{k}^{(2)}_t) - d_t \right) {\boldsymbol{k}^{(1)}_t}^\top + \lambda \boldsymbol{S}^{(1)}_{t-1} \\
\nabla_{\boldsymbol{S}^{(2)}_{t-1}} \mathcal{L}_t &= -\left( (\boldsymbol{S}^{(1)}_{t-1} \boldsymbol{k}^{(1)}_t - \boldsymbol{S}^{(2)}_{t-1} \boldsymbol{k}^{(2)}_t) - d_t \right) {\boldsymbol{k}^{(2)}_t}^\top + \lambda \boldsymbol{S}^{(2)}_{t-1}
\end{aligned}
$$
Gradient descent with step size $\eta$ yields:
$$
\begin{bmatrix} \boldsymbol{S}^{(1)}_t \\ \boldsymbol{S}^{(2)}_t \end{bmatrix} = \begin{bmatrix} \boldsymbol{S}^{(1)}_{t-1} \\ \boldsymbol{S}^{(2)}_{t-1} \end{bmatrix} - \eta \begin{bmatrix} \nabla_{\boldsymbol{S}^{(1)}_{t-1}} \mathcal{L}_t \\ \nabla_{\boldsymbol{S}^{(2)}_{t-1}} \mathcal{L}_t \end{bmatrix}
$$

---

### 5. **Unified Recurrence Form**
Define the **block-matrix state** $\boldsymbol{Z}_t = \begin{bmatrix} \boldsymbol{S}^{(1)}_t \\ \boldsymbol{S}^{(2)}_t \end{bmatrix}$. Then:
$$
\boldsymbol{Z}_t = \boldsymbol{A}_t \boldsymbol{Z}_{t-1} \boldsymbol{B}_t + \boldsymbol{C}_t
$$
where  
- $\boldsymbol{A}_t = \begin{bmatrix} I & 0 \\ 0 & I \end{bmatrix}$ (Identity)  
- $\boldsymbol{B}_t = \begin{bmatrix} I - \eta \lambda I - \eta \boldsymbol{k}^{(1)}_t {\boldsymbol{k}^{(1)}}_t^\top & \eta \boldsymbol{k}^{(1)}_t {\boldsymbol{k}^{(2)}}_t^\top \\ \eta \boldsymbol{k}^{(2)}_t {\boldsymbol{k}^{(1)}}_t^\top & I - \eta \lambda I - \eta \boldsymbol{k}^{(2)}_t {\boldsymbol{k}^{(2)}}_t^\top \end{bmatrix}$  
- $\boldsymbol{C}_t = \begin{bmatrix} \eta d_t {\boldsymbol{k}^{(1)}_t}^\top \\ -\eta d_t {\boldsymbol{k}^{(2)}_t}^\top \end{bmatrix}$

---

### Key Insights
1. **Difference Storage**: Information is stored in the **relative state** $\boldsymbol{S}^{(1)}_{t-1} \boldsymbol{k}^{(1)}_t - \boldsymbol{S}^{(2)}_{t-1} \boldsymbol{k}^{(2)}_t$, not in absolute states.  
2. **Error-Driven Update**: States are adjusted proportionally to the error $e_t = (\boldsymbol{S}^{(1)}_{t-1} \boldsymbol{k}^{(1)}_t - \boldsymbol{S}^{(2)}_{t-1} \boldsymbol{k}^{(2)}_t) - d_t$.  
3. **Regularization**: $\lambda$ ensures stability and prevents state explosion.  
4. **Symmetry**: $B_t$ is symmetric if $\boldsymbol{k}^{(1)}_t {\boldsymbol{k}^{(2)}_t}^\top = (\boldsymbol{k}^{(2)}_t {\boldsymbol{k}^{(1)}_t}^\top)^\top$, which holds for real vectors.  

This formulation extends associative memory to **differential state encoding**, useful for representing contrasts or relationships between entities.

To explain the differential attention mechanism using the two-state associative memory framework, we map the components as follows:

$$
\underbrace{\left(\text{softmax} \left(\boldsymbol{Q}_1 \boldsymbol{K}_1^{\top}\right) \odot \boldsymbol{M}- \lambda \text{ softmax} \left(\boldsymbol{Q}_2 \boldsymbol{K}_2^{\top}\right) \odot \boldsymbol{M}\right)  \boldsymbol{V}}_{\text{Differential Attention}} \equiv \underbrace{\boldsymbol{S}_1 \boldsymbol{k}_1 - \lambda \boldsymbol{S}_2 \boldsymbol{k}_2}_{\text{Two-State Memory Output}}
$$

### Key Mappings:
1. **States as Attention Matrices**:
   - $\boldsymbol{S}_1 \equiv \text{softmax}(\boldsymbol{Q}_1 \boldsymbol{K}_1^{\top}) \odot \boldsymbol{M}$
   - $\boldsymbol{S}_2 \equiv \text{softmax}(\boldsymbol{Q}_2 \boldsymbol{K}_2^{\top}) \odot \boldsymbol{M}$

2. **Keys as Query Projections**:
   - $\boldsymbol{k}_1 \equiv \boldsymbol{V}$ (Values act as input keys)
   - $\boldsymbol{k}_2 \equiv \boldsymbol{V}$

3. **Output**:
   - Differential output: $\boldsymbol{S}_1 \boldsymbol{k}_1 - \lambda \boldsymbol{S}_2 \boldsymbol{k}_2$

---

### Recurrent Formulation
The state evolution follows the two-state memory update:

$$
\begin{bmatrix} \boldsymbol{S}1_t \\ \boldsymbol{S}2_t \end{bmatrix} = \begin{bmatrix} \boldsymbol{A}{1t} & 0 \\ 0 & \boldsymbol{A}{2t} \end{bmatrix} \begin{bmatrix} \boldsymbol{S}1_{t-1} \\ \boldsymbol{S}2_{t-1} \end{bmatrix} + \begin{bmatrix} \boldsymbol{C}{1t} \\ \boldsymbol{C}{2t} \end{bmatrix}
$$

Where for attention:
- $\boldsymbol{A}_{1t}, \boldsymbol{A}_{2t}$: **Decay matrices** (e.g., diagonal gating)
- $\boldsymbol{C}_{1t} = \boldsymbol{q}_{1t} \boldsymbol{k}_{1t}^{\top} \odot \boldsymbol{M}$, $\boldsymbol{C}_{2t} = \boldsymbol{q}_{2t} \boldsymbol{k}_{2t}^{\top} \odot \boldsymbol{M}$  
  (New attention scores for current query/key)

---

### Loss Function (Optimization Objective)
The objective minimizes output error while regularizing states:

$$
\mathcal{L}_t = \frac{1}{2} \left\| (\boldsymbol{S}1_{t-1} \boldsymbol{k}_{1t} - \lambda \boldsymbol{S}2_{t-1} \boldsymbol{k}_{2t}) - \boldsymbol{d}_t \right\|^2 + \frac{\gamma}{2} \left( \|\boldsymbol{S}1_{t-1}\|_F^2 + \|\boldsymbol{S}2_{t-1}\|_F^2 \right)
$$

Where:
- $\boldsymbol{d}_t$: **Target output** (e.g., desired attention output)
- $\gamma$: Regularization strength

---

### Physics of Differential Attention
The system behaves like **coupled oscillators**:
1. **States $\boldsymbol{S}_1, \boldsymbol{S}_2$**: Positions of two particles  
2. **Output difference $\boldsymbol{S}_1 \boldsymbol{k}_1 - \lambda \boldsymbol{S}_2 \boldsymbol{k}_2$**:  
   Measured displacement between particles  
3. **Loss potential**: Spring pulling displacement toward $\boldsymbol{d}_t$  
4. **Regularization**: Friction preventing large state deviations  

$$
\begin{array}{c}
\text{Attention Output} \\
\downarrow \\
\underbrace{\boldsymbol{S}_1 \boldsymbol{k}_1}_{\text{Particle 1}} - \lambda \underbrace{\boldsymbol{S}_2 \boldsymbol{k}_2}_{\text{Particle 2}} \rightarrow \text{Displacement} \\
\downarrow \\
\text{Spring Force: } -\nabla \left[ \frac{1}{2} \| \text{Displacement} - \boldsymbol{d}_t \|^2 \right]
\end{array}
$$

---

### Why This Explains Differential Attention
1. **Contrast Enhancement**  
   The difference $\boldsymbol{S}_1 \boldsymbol{k}_1 - \lambda \boldsymbol{S}_2 \boldsymbol{k}_2$ amplifies **distinctive features** between two attention mechanisms.

2. **Dynamic Gating**  
   Matrices $\boldsymbol{A}_{1t}, \boldsymbol{A}_{2t}$ control memory retention:
   - Diagonal elements $\rightarrow$ input-dependent forgetting
   - Example: $\boldsymbol{A}_{1t} = \text{diag}(\sigma(\boldsymbol{q}_{1t}))$ (sigmoid gating)

3. **Regularization Prevents Overfocus**  
   The $\gamma$-term ensures no single attention state dominates.

---

### Practical Implementation
For a transformer layer:

```python
# Two parallel attention states
\boldsymbol{S}^{(1)} = torch.zeros(B, H, D, D)
\boldsymbol{S}^{(2)} = torch.zeros(B, H, D, D)

for t in range(seq_len):
    q1 = querie\boldsymbol{S}^{(1)}[:, t]  # First attention path
    q2 = querie\boldsymbol{S}^{(2)}[:, t]  # Second attention path
    k = keys[:, t]       # Shared keys
    v = values[:, t]     # Shared values
    
    # Update states with gating
    \boldsymbol{S}^{(1)} = diag_gate1(q1) * \boldsymbol{S}^{(1)} + torch.einsum('b d, b e -> b d e', q1, k) * M[t]
    \boldsymbol{S}^{(2)} = diag_gate2(q2) * \boldsymbol{S}^{(2)} + torch.einsum('b d, b e -> b d e', q2, k) * M[t]
    
    # Differential output
    output = torch.einsum('b d e, b e -> b d', \boldsymbol{S}^{(1)}, v) - λ * torch.einsum('b d e, b e -> b d', \boldsymbol{S}^{(2)}, v)
```

This matches the recurrent form with:
- $\boldsymbol{A}_{1t} = \text{diag\_gate1}(\boldsymbol{q}_{1t})$,  
- $\boldsymbol{C}_{1t} = \boldsymbol{q}_{1t} \boldsymbol{k}^{\top} \odot \boldsymbol{M}$.

The differential output provides **fine-grained contrast** between two attention foci, useful for tasks requiring comparative reasoning (e.g., entailment, object differentiation).




