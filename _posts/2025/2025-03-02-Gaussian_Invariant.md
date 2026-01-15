---
title: Gaussian Invariant
date: 2025-03-02 09:10:08
typora-root-url: ../../allenlu2009.github.io
link: 2021-09-19-Deterministic_Probabilistic
categories:
  - GenAI
tags:
  - Fisher
  - GenAI
  - Information
  - Math-AI
  - Math_AI
  - Score
  - Statistics
---



[[2021-09-19-Deterministic_Probabilistic]]
## Introduction

似乎數學和現實生活有所謂高斯不滅定律。例如機率統計的核心是 central limit theory, 當一堆隨機變數相加，不管這些隨機變數是不是 Gaussian,  最後的和都是 Gaussian.

但是在 ML 的領域的 data 例如 image, text, voice 都是非 Gaussian.   如何 reconcile 這兩者？基本就是利用 deep neural network 可以描述所有 function 的特性。

不過隨機變數有兩個特性，一個是 sample 的特性，另外是 distributin function 的特性。兩者的表現截然不同。像是 time-domain 和 frequency-domain 的關係。**例如 random variable 的和是其 pdf 的 convolution。但反之不成立。**

對於 Gaussian random variable 我們看兩個例子。以下是 Perplexity 的解釋

## Sum of Gaussian RVs is Gaussian (可以忽略)

**不管是 independent 或是 dependent Gaussian RVs.**

The sum of Gaussian random variables is still Gaussian, regardless of whether the variables are independent or dependent. This property arises from the following key principles:

1. **Independence Case**: If $$ X_1, X_2, \ldots, X_n $$ are independent Gaussian random variables with means $\mu_i$ and variances $\sigma_i^2$, their sum $$ S = \sum_{i=1}^n X_i $$ is also Gaussian. The mean and variance of $S$ are:
   $$
   \text{Mean: } \mu_S = \sum_{i=1}^n \mu_i, \quad \text{Variance: } \sigma_S^2 = \sum_{i=1}^n \sigma_i^2
   $$

2. **Dependence Case**: Even if the Gaussian random variables are dependent, their sum remains Gaussian. For a set of jointly Gaussian random variables (i.e., following a multivariate normal distribution), any linear combination of these variables is also Gaussian. The mean and variance of the sum depend on the covariance structure between the variables.

This property is unique to Gaussian distributions and is a key reason for their widespread use in probability, statistics, and signal processing.

## 什麽時候不是 Gaussian?  在算 marginal PDF 的時候！

- **Marginal pdf 是 Gaussian** 的情況： $p(x\vert z)$ 和 $p(z)$ 都是 Gaussian,  而且 $p(x\vert z)$  是 linear function of $z$.

- **Marginal pdf 不是 Gaussian** 的情況： $p(x\vert z)$ 是 Gassian 但是 $p(z)$ 不是 Gaussian.  最簡單的例子是通信。$p(z)$  是 Bernoulli distribution.  此時 $p(x)$ 是一個 mixed Gaussian, 不是 Gaussian! 

- **Marginal pdf 不是 Gaussian** 的情況： $p(x\vert z)$ 和 $p(z)$ 都是 Gaussian,  但是 $p(x\vert z)$  是 non-linear function of $z$，得到的 marginal pdf 就可能是 anything.
	- 在深度學習常用一個 neural network 把 $z$ 轉成 $x$, 例如 auto-encoder.  不過這裏很容易讓人迷惑是 neural network 是一個 deterministic function.   理論上 $p(x\vert z)$  given $z$, 似乎是一個  determinstic value，也就是說 $p(x\vert z)$ 是一個 delta function?  
	- 不過實務上要把 output $x$ given $z$ 視爲一個 random distribution.  比較容易思考是每次產生 output 有一個 sampling 的過程，例如 logits 先產生 softmax (classification function), 加上溫度的影響，最後用 multinomial function 讓每次的采樣 output 都不同！
	- 同樣可以在 output 產生 Gaussian 采樣，這樣 $p(x\vert z)$ 就是一個 Gaussian distribution.  但是 $p(x)$ 不是 Gaussian, 而是 data distribution.

```python
logits, _ = self(idx_cond)
# 加入temperature來處理logits
logits = logits[:, -1, :] / temperature
# apply softmax to convert logits to (normalized) probabilities
probs = F.softmax(logits, dim=-1)
# multinomial函數是用probs作為採樣概率，採樣num_samples個，返回下標，在這裡就是根據概率對所有batch採樣出來
idx_next = torch.multinomial(probs, num_samples=1)
```

---

### 1. Theoretical Framework for Conditional and Marginal Distributions

#### 1.1 Definitions  
- Conditional distribution: $p(x|z)$ represents the distribution of $x$ given $z$. If $p(x|z)$ is Gaussian, then for each $z$, $x \sim \mathcal{N}(\mu(z), \sigma^2(z))$, where $\mu(z)$ and $\sigma^2(z)$ are functions of $z$.  
- Marginal distribution:  
  $$
  p(x) = \int p(x|z)p(z) \, dz,
  $$  
  obtained by integrating the joint distribution $p(x,z)$ over $z$.

---

### 2. Cases Where $p(x)$ Remains Gaussian

#### 2.1 Linear Gaussian Models  
If $x$ and $z$ are jointly Gaussian, the marginal $p(x)$ is Gaussian. For example:  
$$
\begin{aligned}
z &\sim \mathcal{N}(\mu_z, \sigma_z^2), \\
x|z &\sim \mathcal{N}(az + b, \sigma^2),
\end{aligned}
$$  
where $a$, $b$, and $\sigma^2$ are constants. The marginal distribution becomes:  
$$
p(x) = \mathcal{N}\left(a\mu_z + b, \, a^2\sigma_z^2 + \sigma^2\right).
$$

#### 2.2 Multivariate Additive Gaussian Noise  
For multivariate systems:  
$$
\begin{aligned}
\mathbf{z} &\sim \mathcal{N}(\mathbf{\mu}_z, \mathbf{\Sigma}_z), \\
\mathbf{x}|\mathbf{z} &\sim \mathcal{N}(\mathbf{A}\mathbf{z} + \mathbf{b}, \mathbf{\Sigma}_x),
\end{aligned}
$$  
the marginal distribution is:  
$$
p(\mathbf{x}) = \mathcal{N}\left(\mathbf{A}\mathbf{\mu}_z + \mathbf{b}, \, \mathbf{A}\mathbf{\Sigma}_z\mathbf{A}^\top + \mathbf{\Sigma}_x\right).
$$

---

- **Marginal pdf 不是 Gaussian, 像是 neural network 的 auto-encoder
![[Pasted image 20250302091436.png]]
### 3. Cases Where $p(x)$ Is Non-Gaussian

#### 3.1 Discrete or Non-Gaussian $z$  
If $z$ is discrete, $p(x)$ becomes a Gaussian mixture. For example, with $z \in \{0,1\}$:  
$$
p(x) = \frac{1}{2}\mathcal{N}(-1,1) + \frac{1}{2}\mathcal{N}(1,1),
$$  
which is bimodal and non-Gaussian.

#### 3.2 Non-Linear Dependence on $z$  
If $\mu(z)$ or $\sigma^2(z)$ is non-linear in $z$, $p(x)$ is generally non-Gaussian. For $z \sim \mathcal{N}(0,1)$ and $x|z \sim \mathcal{N}(z^2, 1)$:  
$$
p(x) = \int_{-\infty}^\infty \frac{1}{\sqrt{2\pi}} e^{-(x - z^2)^2/2} \cdot \frac{1}{\sqrt{2\pi}} e^{-z^2/2} \, dz,
$$  
which has no closed-form Gaussian solution.

---

### 4. Mathematical Analysis

#### 4.1 Moment-Generating Function (MGF)  
The MGF of $x$ is:  
$$
M_x(t) = \mathbb{E}[e^{tx}] = \mathbb{E}_z\left[\mathbb{E}[e^{tx}|z]\right] = \mathbb{E}_z\left[e^{t\mu(z) + \frac{1}{2}t^2\sigma^2(z)}\right].
$$  
For $x$ to be Gaussian, this requires $\mu(z)$ and $\sigma^2(z)$ to be constants.

---

### 5. Conclusion  
The marginal $p(x)$ is Gaussian if and only if:  
2. $z$ is Gaussian, and  
3. $x$ depends linearly on $z$ with additive Gaussian noise.  

In all other cases, $p(x)$ becomes non-Gaussian. This distinction is critical in probabilistic modeling and inference.  

## Reference

Yang Song, PPDM, ICLR 2021: https://www.youtube.com/watch?v=L9ZegT87QK8&ab_channel=ArtificialIntelligence