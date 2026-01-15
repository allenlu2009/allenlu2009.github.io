---
title: Linear Attention
date: 2024-10-11 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-10-10-Attention_Math]]"
categories:
  - LLM
tags:
  - Attention
  - LLM
  - Transformer
---

[[2023-03-26-Transformer_LLM]] , [[2023-02-18-Attn_All_U_Need_Visual]]

https://www.kexue.fm/archives/7546/comment-page-1: check this article

FAST algorithm: https://arxiv.org/pdf/2009.14794
https://teddykoker.com/2020/11/performers/

Causal and non-causal slides:
https://angeloskath.github.io/data/ml_collective_slides.pdf



## Transformer Series

<img src="/media/image-20241010115242.png" alt="image-20231222085752666" style="zoom:67%;" />


## Takeaways

**Linear attention pros and cons:**
1. **Pros:  linear instead of quadratic storage and computation for long context**
2. **Cons:  1. errors due to base by using approximation or randomized method; 2. Causal mask; 3. Attention dilution (the devil in linear attention);  4. Unbounded gradients (The devil in linear attention)**

**Solution of** 
1. **Use other nonlinear function instead of softmax**
2. **Use chunk-wise + iterative method**
3. **(1) mixed layers of vanilla with linear attention; (2) Forgetting factor, or attention gate**
4. **Normalization**



## KV Cache Issue for Long Context

$O(n^2)$ :  此處 $n$ 是 token length.  如果 $n > 100,000$,  KV cache 變成 dominating computation and storage.  

Transformer total inference resource:

1. Computation, $\text{softmax}(Q K^T) V$: $O(n^2 d)$ for training and the pre-fill.  $O(nd)$ for 1-token generation. $O(n^2 d)$ for $n-$token generation
2. Storage, KV cache:  $O(2nd)$ for $n-$token generation

Linear attention 目標是找到方法 $O(n)$

1. Computation, $Q (K^T V)$: $O(n d^2)$ for training and pre-fill. $O(d^2)$ for 1-token generation. $O(nd^2)$ for $n-$token generation
2. Storage, KV cache:  fixed, not increase with $n, O(d^2)$ for $n-$token generation, even n = 1!
3. Computation and storage linearly grow w.r.t. model size/parameter ($=12\times d^2_{model} \times layer_{num}$ )
<img src="/media/image-20241010133348.png" alt="image-20231222085752666" style="zoom:67%;" />

整個 Attention 的精華就是下式：

$$
\begin{aligned}
\text{Attn} = \text{softmax}(Q K^T)V & 
\end{aligned}
$$
問題是 $Q, K, V \in \mathbb{R}^{n \times d_{model}}$,  如果是 $Q K^T$ 先算，就會變成 $(n \times d_{model}) \times (d_{model} \times n) = n \times n$
如果可以換成

$$
\begin{aligned}
\text{Attn} = Q f(K^T, V) & 
\end{aligned}
$$
就可以 $(d_{model} \times n) \times (n \times d_{model}) = d_{model} \times d_{model}$，  就有機會先整合 $K, V$ 


一個想法是:

$$
\begin{aligned}
\text{Attn} &= \text{similarity}(Q, K)V \\
&= \phi(Q, K) V = \phi(Q) \phi(K) V 
\end{aligned}
$$
這稱為 kernelization.

![[Pasted image 20241205182538.png]]

兩個問題
1. 如何找到 kernelization function
2. 如何處理 causal (auto-regressive) and non-causal (non-autoregressive) attention?


![[Pasted image 20241205182440.png]]

## Kernel Method

這是非常有意思的展開，第一步雖然使用期望值，但沒有任何近似。其實**期望值是嚴謹的數學，是理論推導的好朋友！**。第二步**取樣**才是近似！
如何得到第二步，只要把内積乘開就可以看出來。

$$
e^{q \cdot k}=\mathrm{E}_{\omega \sim \mathrm{N}\left(\omega ; 0,1_d\right)}\left[e^{w \cdot q-\|q\|^2 / 2} \times e^{w \cdot k-\|k\|^2 / 2}\right] \approx \frac{1}{\sqrt{m}}\left(\begin{array}{c}
e^{w_1 \cdot q-||q||^2 / 2} \\
e^{w_2 \cdot q-||q||^2 / 2} \\
\ldots \\
e^{w_m \cdot q-| | q \|^2 / 2}
\end{array}\right) \cdot \frac{1}{\sqrt{m}}\left(\begin{array}{c}
e^{w_1 \cdot k-||k||^2 / 2} \\
e^{w_2 \cdot k-||k||^2 / 2} \\
\ldots \\
e^{w_m \cdot k-||k||^2 / 2}
\end{array}\right)=\tilde{q} \cdot \tilde{k}
$$

上式表示从标准正态分布 $\mathrm{N}\left(\omega ; 0,1_d\right)$ 中采样足够多的 $\omega$ ，然后计算 $e^{w \cdot q-|q|^2 / 2} \times e^{w \cdot k-|k|^2 / 2}$ 的数学期望，结果等于 $e^{q \cdot k}$ 。**上式有非常的好處是均為正值！**  原來的 Kernel method 是用 $\sin, \cos$ 就會有正有負。

上式可以另外表示 (更簡潔) 為

$$
\mathbb{E}_{\omega \sim \mathcal{N}\left(0, \mathbf{I}_d\right)}\left[\exp \left(\omega^{\top} \mathbf{x}-\frac{\|\mathbf{x}\|^2}{2}\right) \exp \left(\omega^{\top} \mathbf{y}-\frac{\|\mathbf{y}\|^2}{2}\right)\right]=\Lambda \mathbb{E}_{\omega \sim \mathcal{N}\left(0, \mathbf{I}_d\right)} \cosh \left(\omega^{\top} \mathbf{z}\right)
$$

$$
\text { where } \Lambda=\exp \left(-\frac{\|\mathbf{x}\|^2+\|\mathbf{y}\|^2}{2}\right) \text { and cosh is hyperbolic cosine. }
$$



实际中 $\omega$ 只能采集有限个，因此采集 $m$ 个并作上述近似。上式将两个 $d$ 维向量的内积的指数 $e^{q \cdot k}$ 转化为两个 $m$ 维向量的内积 $\tilde{q} \cdot \tilde{k}$ ，则对应的自注意力计算为：

$$
\operatorname{Attention}(Q, K, V)_i=\frac{\sum_{j=1}^n\left(\tilde{q}_i \cdot \tilde{k}_j\right) v_j}{\sum_{j=1}^n \tilde{q}_i \cdot \tilde{k}_j}=\frac{\tilde{q}_i \sum_{j=1}^n \tilde{k}_j \cdot v_j}{\tilde{q}_i \sum_{j=1}^n \tilde{k}_j}
$$

此處的 $i, j$ 是 token 的 index, 不是 head,  $i, j \le n$
根据矩阵乘法的结合律，计算复杂度为 $O\left(n^2\right)$ 的 $Q K^T$ 变为计算复杂度为 $O(n)$ 的 $\tilde{K} V^T$ ，从而将自注意力运算线性化。

$\omega_1, \omega_2, \cdot, \omega_m$ 是独立地从标准正态分布 $\mathrm{N}\left(\omega ; 0,1_d\right)$ 中采样得到的。作者指出，若将 $\omega_i$ 进行正交化，能够有效地降低估计方差，提高估计的平均精度。这是因为分布 $\mathrm{N}\left(\omega ; 0,1_d\right)$ 是各向同性的，即在方向上是均匀的，向量正交化使得采样结果更均匀，从而降低估计方差。值得一提的是，上述正交化仅对 $m\le d$ 有效；若 $m>d$，则每 $d$ 个向量进行分组，组内正交化。

作者使用上述机制设计了Performer，当输入序列长度 n 较大时 (比如超过2048)，比标准的Transformer具有明显的速度优势：

![[Pasted image 20241110225514.png]]
作者比较了不同设置下输出的近似程度。采用**正交化采样比独立采样**更有效；本文所提方法比直接**把 $e^{q \cdot k}$ 展开为 $\sin,\cos$ 函数**的形式更精确：

![[Pasted image 20241110225759.png]]

理论上Performer与标准的Transformer可以相互转换。但实验表明Performer直接加载Transformer的权重效果较差，经过微调后能够获得较高的准确率：

![[Pasted image 20241110230955.png]]



### Causal:  Naive computing of Si and Zi 還是 quadratic complexity and sequentially slow!!

![[Pasted image 20241205182920.png]]




### 數學推導

https://0809zheng.github.io/2021/08/12/performer.html

该映射推导如下。注意到 $e^{q \cdot k}$ 可以改写为：

$$
e^{q \cdot k}=e^{\left.\|q\|\right|^2 / 2+||k||^2 / 2-||q-k||^2 / 2}
$$


对 $e^{-||q-k||^2 / 2}$ 做傅里叶变换:

$$
\mathrm{F}\left(e^{-\| q-k| |^2 / 2}\right)=\frac{1}{\sqrt{2 \pi}} \int e^{-||q-k||^2 / 2} e^{-i \omega(q-k)} d(q-k)=\frac{1}{\sqrt{2 \pi}} \int e^{-\|t\|^2 / 2} e^{-i \omega t} d t = \frac{1}{\sqrt{2 \pi}} \int e^{-\left(| | t \|^2+2 i \omega t-\omega^2+\omega^2\right) / 2} d t=e^{-\omega^2 / 2} \frac{1}{\sqrt{2 \pi}} \int e^{-(t+2 i \omega)^2 / 2} d t=e^{-\omega^2 / 2}
$$


再应用傅里叶逆变换：

$$
e^{-\| q-k| |^2 / 2}=\mathrm{F}^{-1}\left(e^{-\omega^2 / 2}\right)=\frac{1}{\sqrt{2 \pi}} \int e^{-\omega^2 / 2} e^{i \omega(q-k)} d \omega
$$


代入原式得：

$$
e^{q \cdot k}=e^{||q||^2 / 2+||k||^2 / 2} \frac{1}{\sqrt{2 \pi}} \int e^{-\omega^2 / 2} e^{i \omega(q-k)} d \omega=\frac{e^{||q||^2 / 2+||k||^2 / 2}}{\sqrt{2 \pi}} \int e^{-\omega^2 / 2+i \omega(q-k)} d \omega
$$


对于上式，若令 $q \rightarrow-i q, k \rightarrow i k$ ，则有:



$$
e^{q \cdot k}=\frac{e^{-||q||^2 / 2-||k||^2 / 2}}{\sqrt{2 \pi}} \int e^{-\omega^2 / 2+\omega(q+k)} d \omega=\int \frac{e^{-\omega^2 / 2}}{\sqrt{2 \pi}} e^{\omega q-\|q\|^2 / 2} \cdot e^{\omega k-\|k\|^2 / 2} d \omega=\mathrm{E}_{\omega \sim \mathrm{N}\left(\omega ; 0,1_d\right)}\left[e^{w \cdot q-\| q| |^2 / 2} \times e^{w \cdot k-\| k| |^2 / 2}\right]
$$




## Some Directions

### 1. Similarity Function

<img src="/media/image-20241010215008.png" alt="image-20231222085752666" style="zoom:67%;" />

### 2. RNN Concept 

<img src="/media/image-20241010215106.png" alt="image-20231222085752666" style="zoom:67%;" />


### 3. Differential Similarity
<img src="/media/image-20241010215151.png" alt="image-20231222085752666" style="zoom:67%;" />



Causal Mask!!
https://arxiv.org/pdf/2009.14794

![[Pasted image 20241027225800.png]]
### Key Question

如何 apply **causal mask** and training????

Causal mask 是 key,  在 transformer model, causal mask 可以很簡單表示。

Linear Attention 基本就是 RNN, 如何 train?  如果要 time step by time step, 基本沒有意義。

還是 training 使用 transformer parallel sequences, 但是 inference 再改成 RNN like?

<img src="/media/image-20241013232254.png" alt="image-20231222085752666" style="zoom:67%;" />

## Source

- Linear Attention 打改变 Transformer 大模型结构垄断 : https://www.bilibili.com/video/BV1V7s9etEmQ/?spm_id_from=333.999.0.0&vd_source=a99fc6374f18662fe559d32fdc3a80cd

- Transformer are RNNs: https://arxiv.org/pdf/2006.16236

- TransNormerLLM:  https://arxiv.org/pdf/2307.14995

- Universal Transformer:  https://arxiv.org/pdf/1807.03819

- Transformer Quality in Linear Time: https://arxiv.org/pdf/2202.10447




## Appendix 

### A:  Use ChatGPT-o1 for Proof

To derive the given equality, we will start by simplifying the left-hand side (LHS) and then show that it equals the right-hand side (RHS) when we define $\mathbf{z} = \mathbf{x} + \mathbf{y}$.

**Step 1: Simplify the LHS**

First, combine the exponents in the LHS:

$$
\begin{align*}
\exp\left(\omega^{\top} \mathbf{x} - \frac{\|\mathbf{x}\|^2}{2}\right) \exp\left(\omega^{\top} \mathbf{y} - \frac{\|\mathbf{y}\|^2}{2}\right) &= \exp\left(\omega^{\top} (\mathbf{x} + \mathbf{y}) - \frac{\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2}{2}\right) \\
&= \Lambda \exp\left(\omega^{\top} (\mathbf{x} + \mathbf{y})\right),
\end{align*}
$$
where $\Lambda = \exp\left(-\frac{\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2}{2}\right)$.

**Step 2: Compute the Expectation**

The LHS becomes:

$$
\mathbb{E}_{\omega \sim \mathcal{N}(0, \mathbf{I}_d)}\left[\Lambda \exp\left(\omega^{\top} (\mathbf{x} + \mathbf{y})\right)\right] = \Lambda \mathbb{E}_{\omega}\left[\exp\left(\omega^{\top} (\mathbf{x} + \mathbf{y})\right)\right].
$$

Since $\omega \sim \mathcal{N}(0, \mathbf{I}_d)$, the moment-generating function of $\omega^{\top} \mathbf{v}$ is:

$$
\mathbb{E}_{\omega}\left[\exp\left(\omega^{\top} \mathbf{v}\right)\right] = \mathbb{E}_{\omega}\left[\exp\left(-\omega^{\top} \mathbf{v}\right)\right] =\exp\left(\frac{1}{2}\|\mathbf{v}\|^2\right).
$$

Applying this to $\mathbf{v} = \mathbf{x} + \mathbf{y}$:

$$
\mathbb{E}_{\omega}\left[\exp\left(\omega^{\top} (\mathbf{x} + \mathbf{y})\right)\right] = \exp\left(\frac{1}{2}\|\mathbf{x} + \mathbf{y}\|^2\right).
$$

So the LHS simplifies to:

$$
\Lambda \exp\left(\frac{1}{2}\|\mathbf{x} + \mathbf{y}\|^2\right).
$$

**Step 3: Simplify the RHS**

Define $\mathbf{z} = \mathbf{x} + \mathbf{y}$. The RHS is:

$$
\Lambda \mathbb{E}_{\omega \sim \mathcal{N}(0, \mathbf{I}_d)}\left[\cosh\left(\omega^{\top} \mathbf{z}\right)\right].
$$

Using the fact that $\cosh(t) = \frac{1}{2}[\exp(t) + \exp(-t)]$ and the symmetry of the normal distribution:

$$
\mathbb{E}_{\omega}\left[\cosh\left(\omega^{\top} \mathbf{z}\right)\right] = \frac{1}{2}\left(\mathbb{E}_{\omega}\left[\exp\left(\omega^{\top} \mathbf{z}\right)\right] + \mathbb{E}_{\omega}\left[\exp\left(-\omega^{\top} \mathbf{z}\right)\right]\right) = \exp\left(\frac{1}{2}\|\mathbf{z}\|^2\right).
$$

Therefore, the RHS simplifies to:

$$
\Lambda \exp\left(\frac{1}{2}\|\mathbf{z}\|^2\right).
$$

**Step 4: Conclude the Equality**

Since $\mathbf{z} = \mathbf{x} + \mathbf{y}$, the LHS and RHS both equal:

$$
\Lambda \exp\left(\frac{1}{2}\|\mathbf{x} + \mathbf{y}\|^2\right).
$$

**Final Result:**

$$
\mathbb{E}_{\omega \sim \mathcal{N}\left(0, \mathbf{I}_d\right)}\left[\exp\left(\omega^{\top} \mathbf{x} - \frac{\|\mathbf{x}\|^2}{2}\right) \exp\left(\omega^{\top} \mathbf{y} - \frac{\|\mathbf{y}\|^2}{2}\right)\right] = \Lambda \mathbb{E}_{\omega \sim \mathcal{N}\left(0, \mathbf{I}_d\right)}\left[\cosh\left(\omega^{\top} (\mathbf{x} + \mathbf{y})\right)\right].
$$

**Answer:**

By simplifying both sides and using properties of the Gaussian, we find that:

$$
\mathbb{E}_{\omega \sim \mathcal{N}(0, \mathbf{I}_d)}\left[\exp\left(\omega^{\top} \mathbf{x} - \frac{1}{2}\|\mathbf{x}\|^2\right) \exp\left(\omega^{\top} \mathbf{y} - \frac{1}{2}\|\mathbf{y}\|^2\right)\right] = \Lambda\, \mathbb{E}_{\omega \sim \mathcal{N}(0, \mathbf{I}_d)}\left[\cosh\left(\omega^{\top} (\mathbf{x} + \mathbf{y})\right)\right]
$$