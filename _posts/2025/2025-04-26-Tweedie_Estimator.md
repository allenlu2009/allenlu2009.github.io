---
title: Three Road Diffusion
date: 2025-03-23 14:10:08
typora-root-url: ../../allenlu2009.github.io
link: 2023-02-08-GenAI_Diffusion
categories:
  - GenAI
tags:
  - GenAI
  - Graph
  - Language
  - Laplacian
  - Math-AI
  - Statistics
---




## Main Reference

[一文解释 经验贝叶斯估计, Tweedie's formula - 知乎](https://zhuanlan.zhihu.com/p/594007789)

https://efron.ckirby.su.domains/papers/2011TweediesFormula.pdf



![[Pasted image 20250427230008.png]]

![[Pasted image 20250427230028.png]]

![[Pasted image 20250318112923.png]]



## Tweedie Estimator

Diffusion 的基本原理：在完全不知道低維 manifold $p_{data}(\mathbf{x})$ 長的什麽樣子，從一個高維 random sample 經過引力場 $\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = -\sigma \mathbf{n}$ 的引導得到低維 image samples，這是蠻神奇的。數學上可以用 Tweedie Estimator 説明。

首先让我们回顾一下**经典贝叶斯**参数估计。首先有一组观测样本 $x=\{x_1,x_2,…,x_n\}$，这一组样本都是从条件概率密度为  $p(x\mid\theta)$  的分布中采样出来的。我们是在已知先验 $p(\theta)$ 参数的前提下，估计目标参数 $\hat{\theta}$。

而**经验贝叶斯** (Empirical Bayes)里，先验分布的参数是没有提前给定的，也是要通过观测数据去估计。

经验贝叶斯方法可以被看成是**分层贝叶斯模型**：在一个两阶段的分层贝叶斯模型中，我们的观测数据 $x=\{x_1,x_2,…,x_n\}$ 是从一组未知的参数 $\theta=\{\theta_1,\theta_2,…,\theta_n\}$ 中根据  $p(x\mid\theta)$  生成的，注意每一个样本 $x_i$ 是只对应参数 $\theta_i$ 的，这里不像经典贝叶斯估计里，对于一个参数我们有多个观测数据，而是对于一个参数我们只有一个观测数据； θ 是从 p(θ|η) 中获取的，其中 p(η) 是非参数的分布，我们也可以直接表示为 p(θ)。目标是估计 θ1,θ2,…,θn。

看一个更具体的以高斯分布为例。我们是先从分布 $N(0,σ^2)$ 采样一组 ${μ_1,μ_2,…,μ_n}$，再分别从分布 $N(μ_i,1)$ 采样一个 $x_i$，构成一组观测 ${x_1,x_2,…,x_n}$，相当于二次采样。


**直观理解就是：经验贝叶斯需要用到辅助的经验信息，待估计参数可以通过其他相关参数进行辅助（因为这些参数是从相同的先验形式），而其他相关参数又是可以通过其他观测数据获得的，这样我们就可以使用来自其他观测数据来改进特定参数的估计性能。


## Tweedie Estimator

Tweedie estimator 原來是一種參數估計方法，其 formulation 如下：
- 假設 $p(\theta)$ 是 **unknown prior**  
- 但是 **conditional probability 是 Gaussian**: $p(x\mid \theta) = N(\theta, \sigma^2)$.  
- 一般 **marginal pdf**, $p(x)=\int p(x \mid \theta)\, p(\theta) \mathrm{d} \theta$, **不是 Gaussian** (除非 prior 是 Gaussian).  
- 目標是從觀察到的 samples $x$ 中推斷未知參數 $\theta$。

儘管 marginal pdf $p(x) = \int p(x \mid \theta)\, p(\theta) \mathrm{d} \theta$ 一般不是高斯分布，但我們可以利用貝葉斯定理來進行有效的參數估計。這樣，我們可以得到後驗分布：

$$
p(\theta \mid x) = \frac{p(x \mid \theta)p(\theta)}{p(x)}
$$
依照 Bayes view $\theta$ 是一個 distribution 而不是一個固定的參數。最好的估計就是計算期望值，也就是 MMSE。以下就是 Tweedie estimator.
$$\begin{aligned}
\theta_{MMSE} = \mathbb{E}_{p(\theta \mid x)}\left[\theta \mid x\right] = \mathbb{E}\left[\theta \mid x\right] = x+\sigma^2 \frac{\mathrm{~d}}{\mathrm{~d} x} \log p(x)\\
\end{aligned}$$
如果 $x$ 是一個高維向量 $\mathbf{x}$ 例如 image,  當然 $\boldsymbol{\theta}$ 也是同樣的高維向量：

$$\boldsymbol{\theta}_{MMSE} = \mathbb{E}_{p(\theta \mid \mathbf{x})}\left[\theta \mid \mathbf{x}\right] = \mathbb{E}\left[\theta \mid \mathbf{x}\right] = \mathbf{x}+\sigma^2 \nabla_{\mathbf{x}} \log p(\mathbf{x})$$
重點是 x 是 observable，而且可以发现 后验期望与先验分布无关 $p(\theta)$，我们没有探究它的具体样子，而仅与边缘分布 $p(x)$ 有关，利用样本估计边缘分布即可。其實我們要估計也不是 $p(x)$， 而是 $p(x)$ 的 score function.   簡而言之，Tweedie formula 有兩個優點：
- 不用探究 prior distribution $p(\theta)$.
- 不用估計 $p(x)$ 最麻煩的 partition function.


## 如何應用？

注意 Tweedie estimator 估計的 $\theta_{MMSE}$ 和 $x$ 基本是一對一。也就是每個 $x_i$ 都有對應的 $\theta_i$.  
當我們有多條獨立同分布的觀測 $x_1, x_2, \dots, x_n$ 時（每個觀測都滿足  $p(x\mid \theta) \sim\mathcal{N}(\theta,\sigma^2)$），
當然我們還需要估計 $p(x)$ 的 score function.  方法如下：

- 如果 sample $x_i$ 數目夠多，可以畫出 $p(x)$ 然後得出 score function.   
- 使用 $x_i, \theta_i, \sigma^2$ 利用 neural network 近似 score function, 此法就是 diffusion 的做法。
- 假設 post distribution $p(x)$ 是 Gaussian or exponential family.  可以直接計算 post distribution 的 score function!

假設 prior distribution $p(\theta) \sim \mathcal{N}(\mu, \tau^2)$, $p(x\mid \theta) \sim\mathcal{N}(\theta,\sigma^2)$, post distribution 
$$p(x) = \int p(x\mid \theta) p(\theta) d\theta = \mathcal{N}(\mu, \sigma^2+\tau^2)$$
$p(x)$ 也是 Gaussian：mean $\mu$,  variance $\sigma^2+\tau^2$
$p(x)$ 的 score function： -$\frac{x-\mu}{\sigma^2+\tau^2}$
$$\theta_i = x - \sigma^2\frac{x-\mu}{\sigma^2+\tau^2} = x_i\frac{\tau^2}{\sigma^2+\tau^2} + \mu\frac{\sigma^2}{\sigma^2+\tau^2}$$ 基本 $\theta$ 的最佳估計就是觀察值, $x$, 和先驗參數, $\tau$,  的綫性平均。Weights 是兩個 Gaussian variance 的比例。這很直覺。如果 $\sigma$ 很小，顯然 $\theta$ 最好用觀察值估計。反之，如果 $\tau$ 很小，最好用先驗值估計。

現實上我們可能沒有 $\mu$, and $\tau$.   當我們有多條獨立同分布的觀測 $x_1, x_2, \dots, x_n$ 時，
$\mu \approx \bar x=\frac1n\sum_{i=1}^n x_i$， $\sigma^2 + \tau^2 \approx \frac{1}{n}\Sigma(x_i - \bar{x})^2)$ 
假設 $\sigma^2$ 是已知，可以同樣用上公式估計每個 $x_i$  對應的 $\theta_i$







## Diffusion vs. Kalman filter
