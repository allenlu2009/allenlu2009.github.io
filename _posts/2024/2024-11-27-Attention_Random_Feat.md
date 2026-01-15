---
title: Attention as Kernel and Random Features
date: 2024-11-27 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-10-10-Attention_Math]]"
categories:
  - LLM
tags:
  - Attention
  - Kernel
  - LLM
  - SVM
  - Transformer
---

[[2023-03-26-Transformer_LLM]] , [[2023-02-18-Attn_All_U_Need_Visual]]

https://www.kexue.fm/archives/7546/comment-page-1: check this article

FAST algorithm: https://arxiv.org/pdf/2009.14794
https://teddykoker.com/2020/11/performers/

https://teddykoker.com/2020/11/performers/

## Attention as SVM Kernel


<img src="/media/image-20241027221621.png" alt="20241027221621" style="zoom:60%;" />


<img src="/media/image-20241027221658.png" alt="20241027221658" style="zoom:60%;" />


### Nystrom ()



### Skyformer (UIUC, 2021)

![[Pasted image 20241127215115.png]]





#### Performer (Google, 2022):  正交和正值

FA: Fast Attention, 就是 kernel 的 ideal, 先做 KV, 再做 Q(KV).  複雜度時 O(n) instead of O(n^2)
OR+:  Orthogonal Random Feature with positive kernel, 如下。
- Orthogonal: 利用 Gram-Schmidt renormalization procedure 產生正交。具體方法是 LR 分解。

### Kernel Method： Fast-Attention

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

上式可以另外表示 (更簡潔) 如下： $\mathbf{z} = \mathbf{x} + \mathbf{y}$

$$
\mathbb{E}_{\omega \sim \mathcal{N}\left(0, \mathbf{I}_d\right)}\left[\exp \left(\omega^{\top} \mathbf{x}-\frac{\|\mathbf{x}\|^2}{2}\right) \exp \left(\omega^{\top} \mathbf{y}-\frac{\|\mathbf{y}\|^2}{2}\right)\right]=\Lambda \mathbb{E}_{\omega \sim \mathcal{N}\left(0, \mathbf{I}_d\right)} \cosh \left(\omega^{\top} \mathbf{z}\right)
$$

$$
\text { where } \Lambda=\exp \left(-\frac{\|\mathbf{x}\|^2+\|\mathbf{y}\|^2}{2}\right) \text { and cosh is hyperbolic cosine. }
$$

比較有趣的是上式基本對 $x, y$ (or $q, k$) 是對稱的。而且如果不考慮 $\Lambda$, 只和 $z$  相關。如何去掉 $\Lambda$?  只需要一開始 normalized $x, y$ (or $q,k$).  


实际中 $\omega$ 只能采集有限个，因此采集 $m$ 个并作上述近似。上式将两个 $d$ 维向量的内积的指数 $e^{q \cdot k}$ 转化为两个 $m$ 维向量的内积 $\tilde{q} \cdot \tilde{k}$ ，则对应的自注意力计算为：

$$
\operatorname{Attention}(Q, K, V)_i=\frac{\sum_{j=1}^n\left(\tilde{q}_i \cdot \tilde{k}_j\right) v_j}{\sum_{j=1}^n \tilde{q}_i \cdot \tilde{k}_j}=\frac{\tilde{q}_i \sum_{j=1}^n \tilde{k}_j \cdot v_j}{\tilde{q}_i \sum_{j=1}^n \tilde{k}_j}
$$

此處的 $i, j$ 是 token 的 index, 不是 head,  $i, j \le n$
根据矩阵乘法的结合律，计算复杂度为 $O\left(n^2\right)$ 的 $Q K^T$ 变为计算复杂度为 $O(n)$ 的 $\tilde{K} V^T$ ，从而将自注意力运算线性化。


### **OR**

$\omega_1, \omega_2, \cdot, \omega_m$ 是独立地从标准正态分布 $\mathrm{N}\left(\omega ; 0,1_d\right)$ 中采样得到的。作者指出，若将 $\omega_i$ 进行正交化，能够有效地降低估计方差，提高估计的平均精度。这是因为分布 $\mathrm{N}\left(\omega ; 0,1_d\right)$ 是各向同性的，即在方向上是均匀的，向量正交化使得采样结果更均匀，从而降低估计方差。值得一提的是，上述正交化仅对 $m\le d$ 有效；若 $m>d$，则每 $d$ 个向量进行分组，组内正交化。

Question: 如何做正交化？Gram-Schemid, How?  any better method?

Gram-Schmidt renormalization.  就是把 $\omega_1, \omega_2, ...\omega_m$  重新 renormalize 成正交 basis.

![[Pasted image 20241111233652.png]]

GS renormalization 實際的計算可以使用 QR decomposition.

$$
\begin{aligned}
&\begin{aligned}
A & =\left[\begin{array}{lll}
\mathbf{x}_1 & \mathbf{x}_2 & \mathbf{x}_3
\end{array}\right] \\
& =\left[\begin{array}{lll}
\mathbf{q}_1 & \mathbf{q}_2 & \mathbf{q}_3
\end{array}\right]\left[\begin{array}{ccc}
r_{11} & r_{12} & r_{13} \\
0 & r_{22} & r_{23} \\
0 & 0 & r_{33}
\end{array}\right] \\
& =\left[\begin{array}{lll}
\mathbf{q}_1 & \mathbf{q}_2 & \mathbf{q}_3
\end{array}\right]\left[\begin{array}{ccc}
\mathbf{q}_1^T \mathbf{x}_1 & \mathbf{q}_1^T \mathbf{x}_2 & \mathbf{q}_1^T \mathbf{x}_3 \\
0 & \mathbf{q}_2^T \mathbf{x}_2 & \mathbf{q}_2^T \mathbf{x}_3 \\
0 & 0 & \mathbf{q}_3^T \mathbf{x}_3
\end{array}\right] \\
& =Q R .
\end{aligned}\\
&\text { 因為 } \mathbf{y}_i=\left\|\mathbf{y}_i\right\| \mathbf{q}_i \text { 且 } \mathbf{q}_i \perp \operatorname{span}\left\{\mathbf{y}_1, \ldots, \mathbf{y}_{i-1}\right\} \text { ，主對角元 } r_{i i} \text { 亦可表示為 }\\
&r_{i i}=\mathbf{q}_i^T \mathbf{x}_i=\mathbf{q}_i^T \mathbf{y}_i=\mathbf{q}_i^T\left(\left\|\mathbf{y}_i\right\| \mathbf{q}_i\right)=\left\|\mathbf{y}_i\right\|
\end{aligned}
$$
Q 矩陣就是 unitary matrix (正交且長度為 1)。



## Spectraformer (UNSW, 2024)

Key: 使用 kernel learning 而不是 random sampling!!   
![[Pasted image 20241127220103.png]]

如何得到 w?
![[Pasted image 20241127223647.png]]


![[Pasted image 20241127222401.png]]

Spectraformer 支持四種方法
1. Random sampling (Monte Carlo):  例如 ORF, SORF
2. Unit Cube sampling: QMC
3. Quadrature (deterministic): SGQ
4. Learner: FastFood.


### DiJiang

![[Pasted image 20241127234356.png]]


### Key Question

如何 apply **causal mask** and training????

Causal mask 是 key,  在 transformer model, causal mask 可以很簡單表示。

Linear Attention 基本就是 RNN, 如何 train?  如果要 time step by time step, 基本沒有意義。

還是 training 使用 transformer parallel sequences, 但是 inference 再改成 RNN like?


<img src="/media/image-20241013232254.png" alt="20241013232254" style="zoom:60%;" />

## Source

- Linear Attention 打改变 Transformer 大模型结构垄断 : https://www.bilibili.com/video/BV1V7s9etEmQ/?spm_id_from=333.999.0.0&vd_source=a99fc6374f18662fe559d32fdc3a80cd

- Transformer are RNNs: https://arxiv.org/pdf/2006.16236

- TransNormerLLM:  https://arxiv.org/pdf/2307.14995

- Universal Transformer:  https://arxiv.org/pdf/1807.03819

- Transformer Quality in Linear Time: https://arxiv.org/pdf/2202.10447



