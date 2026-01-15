---
title: Math AI - Score Matching is All U Need for Diffusion
date: 2025-02-02 23:10:08
typora-root-url: ../../allenlu2009.github.io
categories:
  - GenAI
tags:
  - Diffusion
  - Fisher
  - GenAI
  - Information
  - Math-AI
  - Math_AI
  - Score
---





## Yang Song 的 interview: "History of Diffusion" 提到
- 爲什麽 focus on score matching 的關鍵是: MCMC sampling of $p_{data}(\mathbf{x})$ 太慢了。因爲只有 accept and reject.   Score matching 因爲有指引所以快的多。
- **很多人都說 score function 有非常多挑戰不可行。他也承認，但是初生之犢不怕虎。最後加上 gaussian noise 什麽問題都解決了！**
- Score matching 雖然快，缺點是比較注重在 local structure.   MCMC 因爲有 partition function, 比較有全局觀。
- 發展 SDE (continuous method) 是被 DDPM 刺激，可以結合 score matching and DDPM.
- A remarkable result from Anderson (1982) states that the reverse of a diffusion process is also a diffusion process, running backwards in time and given by the reverse-time SDE:
- Yang 也 surprise reverse SDE 的 close form (Anderson) 而且包含 score function!
- 很自然從 ODE based on Fokker-Planck, 原來是爲了計算 likelihood, 沒想到可以加速。
- 進一步 neural ODE 得到 consistency model, 2 step 就可以得到 comparable quality image.  同時可以解釋 flow model.  一般 flow model 使用 reciprocal function, 但是也可以用 ODE 于 reciprocal function!
- UNET 是最適合 score function.   
- No surprise of stable diffusion (Unet on latent).  DiT 才是真正 transformer based.
- Consistency model 是另一個 AR, Diffusion 之外的 generative method.  很看好


但是加上“適當”的 noise, 反而很容易產生 image sample (diffusion method)。

其實和 noise scheduling 是同一件事。


## Langevin Dynamics

Score function 的一個非常特別的應用！就是用來產生 $p(\mathbf{x})$ 的 fair samples，稱為 Langevin dynamics.  Given a well-behaved probability function $P(\mathbf{x})$, we aim to draw i.i.d sample from it.  這是一個 general results, diffusion method 被用來產生 image, 如下圖。

我們後面會用 (a) Fokker-Planck equation 驗證 Gaussian distribution 符合 Langevin dynamics; (b) 用 Reverse-Time SDE 驗證 general distribution (不過好像差一個 $\sqrt{2}$?).   **Langevin dynamics 可以視為 reverse SDE 的 stationary special case.**

![[Pasted image 20250201223711.png]]

上圖公式的 $\mathbf{x}_t$，$\mathbf{w}_t$ 是對時間的微分，不過很容易和 $\mathbf{x}_t$，$\mathbf{w}(t)$ 混淆。所以我改成以下更常見的公式：explicit 微分 by $d(.)$ and $dt$，$\mathbf{x}_t = \mathbf{x}_t$，$\mathbf{w}_t = \mathbf{w}(t)$。

**注意在 Langevin dynamics score function (log-likelihood 梯度) 中的 $p(\mathbf{x})$  本身和時間無關，其實可以視為 stationary distribution。但是梯度之後的 $\mathbf{x}$ 要換成 $\mathbf{x}_t$.  我們接下來會看一些例子。**

$$
d\mathbf{x}_t = \nabla_{\mathbf{x}} \log p(\mathbf{x}) \, dt + \sqrt{2} \, d\mathbf{w}_t
$$

如果沒有 $\mathbf{w}_t$ 基本 $\mathbf{x}_t$ 會直接趨向 maximum likelihood of $p(\mathbf{x})$。這是 optimization, 而不是產生 "fair samples".  但是加上 $d\mathbf{w}_t$ 的 white noise, $d\mathbf{w}_t \sim N(0, dt)$, (i.e. $\mathbf{w}_t\sim N(0, t)$ 是隨時間變大的 random walk). 此時 $\mathbf{x}_t$ 會產生 "fair samples" of $p(\mathbf{x})$.

> Q：因為有 $d \mathbf{w}_t$，這裏產生的 "fair samples" 是 noisy samples？ 
> A: NO, 不是 noisy sample.  因為 $p(\mathbf{x}_t)$  本來就是一個 distribution.  $d\mathbf{w}_t$ 會讓 $\mathbf{x}_t$ 遍歷所有的 distribution, 而不是 noise.
> 
> 所謂的 noisy sample 是指在原來的 $p_{data}(\mathbf{x})$ distribution 故意加上額外的 noise, i.e. $\mathbf{x}+\Sigma_t \boldsymbol{\epsilon}$, where $\boldsymbol{\epsilon}\sim N(0, I)$.  這時需要做 noise scheduling $\Sigma_t$ 才能得到原來的 image sample.  我們在下一節會討論 noise scheduling in a more general setting.

## Gaussian Samples on Langevin Dynamics

假設 $p(\mathbf{x})$ 是 Gaussian distribution $N(0, I)$,  score function 為 $\nabla_{\mathbf{x}} \log p(\mathbf{x}) = -\mathbf{x}$.  上式可以簡化成

$$
d\mathbf{x}_t = -\mathbf{x}_t \, dt + \sqrt{2} \, d\mathbf{w}_t
$$

- 如果忽略 $d\mathbf{w}_t$，$d\mathbf{x}_t = -\mathbf{x}_t \, dt$，也就是 $\mathbf{x}_t$ 越大就會有更大的負作用力拉回原點。解 ODE 微分方程：可以得到 $\mathbf{x}_t = \mathbf{x}_{0}\exp(-t)$，也就是任何 random initial $\mathbf{x}_0$ 都會回到原點。這是一個標準一階 maximum likelihood optimizer!   不過不是想得到的 $N(0, I)$.
- 相反，如果忽略 score function, $d\mathbf{x}_t = \sqrt{2} \, d\mathbf{w}_t$, 這是一個標準的 random walk.  也就是 $\mathbf{x}_t\sim N(0, 2t)$.
	- Why $\sqrt{2}$?  可以從下面驗證得出。如果沒有 $\sqrt{2}$，最後的 distribution 的 variance 會變小，反之則變大。這符合直覺。
- 當然上式的正確解不是兩者相加，因為 score function 會指引回家的路！正確的解是要用  [Fokker–Planck equation - Wikipedia](https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation) 或是 special case: [Ornstein–Uhlenbeck process - Wikipedia](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) 如下圖。
- 帶入 $a=\theta=1$ and $\sigma=\sqrt{2}$， 可以得到穩態解 $N(0, I)$。這證明上圖的方法可以產生 fair sampes.
$$
p_{ss}(\mathbf{x}) = \frac{1}{\sqrt{2\pi}} e^{-\frac{\mathbf{x}^2}{2}}
$$

#### Langevin dynamics on general Gaussian distribution
$p(\mathbf{x}) \sim N(\mu, \sigma^2)$,  score function 為 $\nabla_{\mathbf{x}} \log p(\mathbf{x}) = -\frac{\mathbf{x}-\mu}{\sigma^2}$.  上式可以簡化成

$$
d\mathbf{x}_t = -\frac{\mathbf{x}_t-\mu}{\sigma^2} \, dt + \sqrt{2} \, d\mathbf{w}_t
$$
明顯 $\mu$ 只是 offset, 可以用變數代換移除最後再加回。 $a=\Sigma^{-1}$ and $\sigma=\sqrt{2}$. 

$$
p_{ss}(\mathbf{x}) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(\mathbf{x}-\mu)^2}{2\sigma^2}}
$$
- $\sqrt{2}$ 的常數 on $d\mathbf{w}_t$ 是必要的。


## OU Process 一般解

![[Pasted image 20250417003916.png]]

如果考慮 general OU process 如下 ($\theta=a, D=\frac{\sigma^2}{2}$).  初始 pdf 可以是任何 distribution, 不一定是 Gaussian.
$$
d\mathbf{x}_t = -\theta \,\mathbf{x}_t \, dt + \sigma \, d\mathbf{w}_t
$$
首先計算 transition probability.  因為加的是 Gaussian noise $d\mathbf{w}_t \sim N(0, dt)$, 我們應該不會驚訝 transition (conditional) probability 是 Gaussian distribution. 

$$\begin{aligned}
P\left(x, t \mid x^{\prime}, t^{\prime}\right)&=\sqrt{\frac{\theta}{2 \pi D\left(1-e^{-2 \theta\left(t-t^{\prime}\right)}\right)}} \exp \left[-\frac{\theta}{2 D} \frac{\left(x-x^{\prime} e^{-\theta\left(t-t^{\prime}\right)}-\mu\left(1-e^{-\theta\left(t-t^{\prime}\right)}\right)\right)^2}{1-e^{-2 \theta\left(t-t^{\prime}\right)}}\right]\\
&= N \left( x^{\prime} e^{-\theta\left(t-t^{\prime}\right)}+\mu (1 - e^{-\theta (t - t')}), \quad \frac{D}{\theta}(1 - e^{-2\theta(t - t')})  \right)
\end{aligned}$$
### Transition probability 極端 cases:
- $t\to t'$: $P\left(x, t \mid x^{\prime}, t^{\prime}\right) \to N ( x^{\prime}, 0) = \delta(x-x')$
- $t\to \infty$: $P\left(x, t \mid x^{\prime}, t^{\prime}\right) \to N ( \mu, \frac{D}{\theta}) = N ( \mu, \frac{\sigma^2}{2\theta})$

![[Pasted image 20250417232829.png]]

### **Transition probability 一般解**
因為 from $x_{t'} = x'$ to $x_t = x$ has a Gaussian distribution, 我們只需要計算 mean and variance:

- **Mean:**  
  $$
  \mathbb{E}[x_t \mid x_{t'} = x'] = x' e^{-\theta (t - t')} + \mu \left(1 - e^{-\theta (t - t')}\right) = \mu + (x'-\mu)e^{-\theta (t - t')}
  $$

- **Variance:**  
  $$
  \text{Var}[x_t \mid x_{t'} = x'] = \frac{D}{\theta} \left(1 - e^{-2\theta (t - t')}\right)
  \quad \text{with } D = \frac{\sigma^2}{2}
  $$
- Mean 的 time constant 是 $\frac{1}{\theta}$ 應該很直覺。
- Variance 的 time constant 是 $\frac{1}{2\theta}$ 比較特別。

### **計算 marginal distribution of $\mathbf{x}_t$**

如果 initial distribution 是 Gaussian, transition probability distribution 是 Gaussian, final distribution 也是 Gaussian.  所以只需要計算最後的 mean and variance.  如果 initial distribution 不是 Gaussian, 可以用積分 with Gaussian kernel.

假設 **initial distribution** $x_{t'} \sim N(0, S)$, 如果 $S$ 非常大，可以視爲 uniform distribution over 整個空間。 The **final distribution** $x_t$ at time $t > t'$ can be derived using the transition probability. 

If $x_{t'} \sim N(0, S)$, then marginalizing over $x'$ yields the unconditional distribution of $x_t$:

1. The **mean** becomes:
   $$
   \mathbb{E}[x_t] = \mathbb{E}_{x'}[\mathbb{E}[x_t \mid x']] = \mathbb{E}[x' e^{-\theta (t - t')} + \mu (1 - e^{-\theta (t - t')})] = \mu (1 - e^{-\theta (t - t')})
   $$
   (since $\mathbb{E}[x'] = 0$)

2. The **variance** becomes:
   $$
   \text{Var}(x_t) = \mathbb{E}_{x'}[\text{Var}(x_t \mid x')] + \text{Var}_{x'}[\mathbb{E}(x_t \mid x')]
   $$
   First term:
   $$
   \frac{D}{\theta}(1 - e^{-2\theta (t - t')})
   $$
   Second term:
   $$
   \text{Var}(x' e^{-\theta(t - t')}) = S e^{-2\theta(t - t')}
   $$

   So total variance:
   $$
   \text{Var}(x_t) = \frac{D}{\theta}(1 - e^{-2\theta (t - t')}) + S e^{-2\theta(t - t')} = \frac{D}{\theta} + (S-\frac{D}{\theta})e^{-2\theta(t - t')}
   $$
不管初始的 variance $S$ 有多大，都會以 $\frac{1}{2\theta}$  的 exponential time constant decay,  反而 $\frac{D}{\theta}$  會以 exponential time constant 增加，變成最後的 variance $\frac{D}{\theta}$!

如果 $S = D/\theta=\sigma^2/2\theta$, variance preserve

### **Final Result:**

$$
x_t \sim N \left( \mu (1 - e^{-\theta (t - t')}), \quad \frac{D}{\theta}(1 - e^{-2\theta(t - t')}) + S e^{-2\theta(t - t')} \right)
$$

This gives the full distribution of $x_t$ evolved from an initial Gaussian $N(0, S)$ under the OU process. 


當 $t \to +\infty$ 得到穩態解。
$$
x_{\infty} \sim N \left( \mu , \frac{D}{\theta}\right)
$$

$$
P\left(x, t_{\infty} \right)=\sqrt{\frac{\theta}{2 \pi D}} \exp \left[-\frac{\theta}{2 D}{\left(x-\mu\right)^2}\right] = \sqrt{\frac{\theta}{\pi \sigma^2}} \exp \left[-\frac{\theta}{\sigma^2}{\left(x-\mu\right)^2}\right]
$$

基本是從 $x_{t'} \sim N(0, S)$  收斂到 $x_{\infty} \sim N \left( \mu , \frac{D}{\theta}\right) = N(0, I)$  的過程。
注意這是 marginal pdf, 不是前面說的 transition pdf.  

## Noise Scheduling Diffusion Process

前面我們只討論如何產生 fair sample, 但沒有 noise scheduling!
要做 noise scheduling, 需要更 general SDE 如下：
$$
d \mathbf{x}_t={\boldsymbol{f}}(\mathbf{x}_t, t)\, d t+g(t)\, d \mathbf{w}_t,\quad \text{ with } d \mathbf{w}_t \sim N(0, d t)
$$
第一項是 $f \in R^{d\times d}$ deterministic drift term , 第二項 $g(t)\, \mathbf{w}_t$ 稱為 stochastic diffusion term, $g(t)$ 是 scalar.  

這個隨機微分方程對時間的關係表現在兩方面：
- $\mathbf{x}_t = \mathbf{x}_t$ 隨時間在空間中移動，稱爲 (random) sample trace.  
![[Pasted image 20250419200722.png]]
另外 $\mathbf{x}_t =\mathbf{x}_t$  對應的 (determinstic) pdf: $p(\mathbf{x}_t) = p(\mathbf{x}_t)=p_t(\mathbf{x})$ 隨時間變化。例如上面從 $x_{t'} \sim N(0, S)$  收斂到 $x_{\infty} \sim N \left( \mu , \frac{D}{\theta}\right)$  的過程。
$$
x_t \sim N \left( \mu (1 - e^{-\theta (t - t')}), \quad \frac{D}{\theta}(1 - e^{-2\theta(t - t')}) + S e^{-2\theta(t - t')} \right)
$$
![[FokkerPlanck.gif]]


### Reverse SDE

**上式 SDE 有一個 reverse 的 SDE.  對應反向的 sample trace 和 pdf 的演化。這是魔法所在！**
$$
d \mathbf{x}_t=[{\boldsymbol{f}}(\mathbf{x}_t, t)-g^2(t)\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)]\, d t+g(t)\, d \mathbf{w}_t,\quad \text{ with } d \mathbf{w}_t \sim N(0, d t)
$$

**為什麼要做 reverse SDE?   因為對於一些 data distribution, 例如 image, voice, (1) 找到 $p(\mathbf{x}_{data})$; (2) 再產生 image sample 非常困難。

但是用反向 SDE,  從 random sample 經過 score function 的指引可以直接 (2) 產生 sample 而不用 (1)。以上就是實例。我們再整理一下：

- $f(\mathbf{x}_t, t) = 0$； $g(t) = \sqrt{2}$
- Forward: $d \mathbf{x}_t =  \sqrt{2} \, d\mathbf{w}_t$
- Backward: $d \mathbf{x}_t = -2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) dt +  \sqrt{2}\,d\mathbf{w}_t$
or 
- $f(\mathbf{x}_t, t) = 0$； $g(t) = 1$
- Forward: $d \mathbf{x}_t = d\mathbf{w}_t$
- Backward: $d \mathbf{x}_t = - \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) dt +  \,d\mathbf{w}_t$

**以上和前面 Langevin dynamics 的 “fair sample" 假説看起來非常相似，但有不同的地方**
- Score function 是負的！是因爲 $dt \to -dt$?  yes, $dt$ 是負無窮小 time step
- 特別 (也是有特別有用) 的部分是 score function 對應的 pdf 是隨時間變化 $p_t({\mathbf{x}_t})$！而不是 “fair sample" 的 $p(\mathbf{x})$.
- 我的解釋：”fair sample" 對應的是終極態 ($t\to\infty$) 的 $p(\mathbf{x})$ 的 score function.  所以在 reverse SDE 雖然是用暫態的 $p_t(\mathbf{x})$ 的 score function, 但隨著 score function 的指引，Gaussian noise 的部分會越來越小，最後還是會趨近 $p(\mathbf{x})$ 的 score function。
- 當 $t > T$  (足夠大) 已經非常趨近終極態，$p_t(\mathbf{x})\to p_{\infty}(\mathbf{x})=p(\mathbf{x})$。此時的 (近似) 終極態其實是動態的終極態，就像一個 Markov Chain， $t$ 的增加會遍歷  $p(\mathbf{x})$ distribution 但是不會改變 $p(\mathbf{x}_t) = p(\mathbf{x})$.
- $d\mathbf{w}_t$ 少了 $\sqrt{2}$，這似乎會造成最後 $p(\mathbf{x}_t)$ 無法返回 $p(\mathbf{x})$，而是返回一個 variance 只有原來 1/2 的 distribution?  (???) 還是因為在 forward path 會造成同樣的 noise?  所以兩者相加得到 $\sqrt{2}$? 


這也帶到我們接下來的主題，如果沒有  $p(\mathbf{x})$ 的 score function 怎麽辦？
答案已經呼之欲出。就是 forward path 的加 noise!   只是可以更複雜一點用 $g(t)$

### How to Find Score Function? 

上面最大的問題是需要 score function.  在例子中因為是 Gaussian distribution, 可以直接得到 close form score function.  但是對於一般 data distribution，基本和直接找到 $p_{data}(\mathbf{x})$ 一樣困難。不過一個關鍵的好處是 score function 比較 pdf 少了 partition factor :)


Score matching using neural network $\mathbf{s}_{\theta}$
$$\min_{\theta} \mathbb{E}_{p_{data}(\mathbf{x})}(\| \mathbf{s}_{\theta}(\mathbf{x}) - \nabla_\mathbf{x} \log p_{data}(\mathbf{x}) \|^2)$$

#### Naive Version (Not Working)
上式一個問題是我們不知道 $p_{data}(\mathbf{x})$，但是對與 $p_{data}(x)$ 的期望值可以用 sample 平均近似。經過複雜的置換，可以把 $p_{data}(\mathbf{x})$ 從目標函數換掉。但還留在期望值。
$$
\min_{\theta} \mathbb{E}_{p_{\text {data }}(\mathbf{x})}\left[\operatorname{tr}\left(\nabla_{\mathbf{x}} \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})\right)+\frac{1}{2}\left\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})\right\|_2^2\right]
$$

Training Issue: too big (nxn)
Sampling issue: not working!

**Based on Yang Song (inventor of score matching): 加上 gaussian noise 什麽問題都解決了！**

## Score Matching (精華)

**NCSN - Noise Conditioned Score Network**
The training objective was proved equivalent to the following:  [reference: P. Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661–1674, 2011.]

$$\begin{aligned}
&\min_{\theta} \mathbb{E}_{q_\sigma(\tilde{\mathbf{x}})}\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma)-\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}})\right\|_2^2\right]\\
=&\min_{\theta} \mathbb{E}_{q_\sigma(\tilde{\mathbf{x}} , \mathbf{x})}\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma)-\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})\right\|_2^2\right]\\
=&\min_{\theta} \mathbb{E}_{p_{\text {data }(\mathbf{x})}\,}\mathbb{E}_{q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})}\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma)-\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})\right\|_2^2\right]
\end{aligned}$$

The optimal score network that minimizes Eq. satisfies $\mathbf{s}_{\boldsymbol{\theta}^*}(\tilde{\mathbf{x}}, \sigma)=\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}})$. 
However, $\mathbf{s}_{\boldsymbol{\theta}^*}(\tilde{\mathbf{x}}, \sigma)=\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}}) \approx \nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})$ is true only when the noise is small enough such that $q_\sigma(\mathbf{x}) \approx p_{\text {data }}(\mathbf{x})$.

最簡單是 additive Gaussian noise，其表示式為 $\tilde{\mathbf{x}} = \mathbf{x} + \sigma \mathbf{n}$，此處 $\mathbf{n} \sim N(0, I)$  **high-dimension** noise.  上式可以簡化成 noise prediction or denoise.  
$$\mathbf{s}_{\boldsymbol{\theta}^*}(\tilde{\mathbf{x}}, \sigma)=\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}}) = -\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^2}$$
其實這是 Tweedie's estimator, 基本是 MMSE estimator!

因爲 $\sigma$ 會接近 0, 顯然放在分母會有數值收斂的問題，因此可以乘 $\sigma^2$.
**實際做法不是 predict score function, 而是 predict noise: 
$$\sigma^2\mathbf{s}_{\boldsymbol{\theta}^*}(\tilde{\mathbf{x}}, \sigma)=\sigma^2\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}}) = -(\tilde{\mathbf{x}}-\mathbf{x})$$
上式可以改寫如下，主要是爲了後面的 noise scheduling！
$$\begin{aligned}
&\min_{\theta} \mathbb{E}_{p_{\text {data }(\mathbf{x})}\,}\mathbb{E}_{q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})}\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma)-\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})\right\|_2^2\right]\\
=&\min_{\theta} \sigma^2 \, \mathbb{E}_{p_{\text {data }(\mathbf{x})}\,}\mathbb{E}_{q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})}\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma)-\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})\right\|_2^2\right]\\
=&\min_{\theta}  \, \mathbb{E}_{p_{\text {data }(\mathbf{x})}\,}\mathbb{E}_{q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})}\left[\sigma^{-2}\left\|\sigma^2 \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma)-\sigma \mathbf{n}\right\|_2^2\right]\\
=&\min_{\theta} \, \mathbb{E}_{p_{\text {data }(\mathbf{x})}\,}\mathbb{E}_{q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})}\left[\sigma^{-2}\left\| D_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma)-\mathbf{x}\right\|_2^2\right]
\end{aligned}$$
其中
$$D_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma) = \tilde{\mathbf{x}}+\sigma^2\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma)\approx\tilde{\mathbf{x}}+\sigma^2\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}}) = \mathbf{x}$$
此處因爲 $\sigma^2$ 是個常數，所以乘上常數沒有影響。但是後面的 noise scheduling 可以看出 weighting based on noise 的大小。
爲什麽乘上 $\sigma^2$？因爲 $$\mathbf{s}_{\boldsymbol{\theta}^*}(\tilde{\mathbf{x}}, \sigma)\approx\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}}) = -\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^2} = -\frac{\sigma \mathbf{n}}{\sigma^2} \propto \frac{1}{\sigma}$$
而且有一個 L2 norm 平方項，所以乘上 $\sigma^2$ to compensate the scale.  這個在後面 noise scheduling 可以清楚看出。

利用 neural network 近似 time-varying $\mathbf{s}_{\theta} \approx \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$. 

### NCSN Noise Scheduling - Discrete Time

**Training**

$\tilde{\mathbf{x}} = \mathbf{x} + \sigma_i \mathbf{n}$.  使用 $\sigma_{min} = \sigma_1 < \cdots < \sigma_N = \sigma_{max}$
$$
\boldsymbol{\theta}^* = \underset{\boldsymbol{\theta}}{\arg \min} \sum_{i=1}^N \sigma_i^2 \, \mathbb{E}_{p_{\text{data}}(\mathbf{x})} \, \mathbb{E}_{p_{\sigma_i}(\tilde{\mathbf{x}} \mid \mathbf{x})} \left[ \left\| \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma_i) - \nabla_{\tilde{\mathbf{x}}} \log p_{\sigma_i}(\tilde{\mathbf{x}} \mid \mathbf{x}) \right\|_2^2 \right]
$$
**因爲乘以 $\sigma_i^2$ ，所以每一項基本都 normalized to equal weights，都一樣重要！**

**Sampling (2 loops，$m$ and $i$)**

Method 1: 利用 Annealed (noise scheduling) Langevin dynamics
$$
d\tilde{\mathbf{x}}_t = \nabla_{\tilde{\mathbf{x}}} \log p(\tilde{\mathbf{x}}) \, dt + \sqrt{2} \, d\mathbf{w}_t
$$
以上是 continuous 表示，實務上 Discrete version
$$
\mathbf{x}_i^m = \mathbf{x}_i^{m-1} + \epsilon_i \, \mathbf{s}_{\boldsymbol{\theta}^*}(\mathbf{x}_i^{m-1}, \sigma_i) + \sqrt{2 \epsilon_i} \, \mathbf{z}_i^m, \quad m=1,2,\cdots, M
$$

所以 sample algorithm: outer loop on $\sigma_i$,  inner loop on $t$ 
![[Pasted image 20250425160532.png]]
### DDIM Noise Scheduling - Discrete Time

**Training:**
$$
\boldsymbol{\theta}^* = \underset{\boldsymbol{\theta}}{\arg \min} \sum_{i=1}^N (1-\alpha_i) \, \mathbb{E}_{p_{\text{data}}(\mathbf{x})} \, \mathbb{E}_{p_{\alpha_i}(\tilde{\mathbf{x}} \mid \mathbf{x})} \left[ \left\| \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, i) - \nabla_{\tilde{\mathbf{x}}} \log p_{\alpha_i}(\tilde{\mathbf{x}} \mid \mathbf{x}) \right\|_2^2 \right]
$$
**Sampling: (only 1 loop, on $i$)**

$$
\mathbf{x}_{i-1} = \frac{1}{\sqrt{1-\beta_i}} \left( \mathbf{x}_i + \beta_i \, \mathbf{s}_{\boldsymbol{\theta}^*}(\mathbf{x}_i, i) \right) + \sqrt{\beta_i} \, \mathbf{z}_i, \quad i = N, N-1, \cdots, 1
$$


## Continuous-time SDE
![[Pasted image 20250423231507.png]]


![[Pasted image 20250201224437.png]]
![[Pasted image 20250201233020.png]]

![[Pasted image 20250202115530.png]]


第一項是 $f \in R^{d\times d}$ deterministic drift term , 第二項 $g(t)\, \mathbf{w}_t$ 稱為 stochastic diffusion term, $g(t)$ 是 scalar.  如果 forward SDE 是 diffusion process,  reverse SDE 也是 diffusion process.

**Forward SDE: for training.**
$$
d \mathbf{x}_t={\boldsymbol{f}}(\mathbf{x}_t, t)\, d t+g(t)\, d \mathbf{w}_t,\quad \text{ with } d \mathbf{w}_t \sim N(0, d t)
$$

**Reverse SDE: for sampling.**  下式的 $dt$ 是**負**無窮小 time step.
$$
d \mathbf{x}_t=[{\boldsymbol{f}}(\mathbf{x}_t, t)-g^2(t)\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)]\, d t+g(t)\, d \mathbf{w}_t,\quad \text{ with } d \mathbf{w}_t \sim N(0, d t)
$$

**Equivalent Fokker-Planck ODE**
$$
d \mathbf{x}_t=[{\boldsymbol{f}}(\mathbf{x}_t, t)-\frac{1}{2} g^2(t)\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)]\, d t
$$
### Training by Score Matching

Continuous SDE 基本形式和 NCSN and DDPM 一樣如下式，但從 discrete 變成 continuous，和對應不同的 $\mathbf{x}_t$ (forward) **noise diffusion 的方式和速度，造成不同的 transition probability 和 $\lambda(t)$.  也就是説，forward diffusion 控制 noise/denoise scheduling**.   

**在宋飏的論文建議三種 forward diffusion:  (1) VE, Variance Explode, continuous NCSN; (2) VP, Variance Preserve, continuous DDPM; (3) sub-VP,  是新的 continuous SDE.** 

$$
\boldsymbol{\theta}^* = \underset{\boldsymbol{\theta}}{\arg \min}\, \mathbb{E}_t \left\{ \lambda(t) \, \mathbb{E}_{\mathbf{x}_0} \, \mathbb{E}_{\mathbf{x}_t \mid \mathbf{x}_0} \left[ \left\| \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log p_{0t}(\mathbf{x}_t \mid \mathbf{x}_0) \right\|_2^2 \right] \right\}
$$
$$
\lambda(t) \propto 1 / \mathbb{E}\left[\left\|\nabla_{\mathbf{x}_t} \log p_{0 t}(\mathbf{x}_t \mid \mathbf{x}_0)\right\|_2^2\right] .
$$
Score matching 就是訓練一個 neural network: $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)$ 近似 score function: $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$.
這個 optimization (分成兩個期望值) 等價於近似 transition probability score function: $\nabla_{\mathbf{x}_t} \log p_{0t}(\mathbf{x}_t \mid \mathbf{x}_0)$.
$\lambda(t)$ 的選擇就是讓每一項 $t$ 都有接近一樣的 loss contribution.
最外面還有一個期望值 $\mathbb{E}_t$ 在 training 也很重要：$0 < t \le T$ 一般用 uniform distribution.  

#### Continuous Transition Probability in Score Function

Trasition probability 是 Gaussian (因爲 inject Gaussian noise), 有 close form: $p_{0t}(\mathbf{x}_t \mid \mathbf{x}_0) = N(\boldsymbol{\mu}_t, \boldsymbol{\sigma}^2_t) = N(\boldsymbol{\mu}_t, {\sigma}^2_t\mathbf{I})$, 因爲是 isotropic (homogenous) Gaussian.

Forward SDE from 0 to $t$, 所以Gaussian 的 
- $\boldsymbol{\mu}_t = mean(\mathbf{x}_t)$ with i.c. $\boldsymbol{\mu}_0 = \mathbf{x}_0$ 
	- 方法: 解 **noise-free ODE** 方程 $dx = f(x,t) dt$ with i.c. $x_0$.  此處可以忽略 zero-mean $dw$.  
	- VE ODE: $f=0$, $\boldsymbol{\mu}_t = \mathbf{x}_0$
	- VP ODE: $f=-\frac{1}{2}\beta(t)x$,  $d{x} = -\frac{1}{2} \beta(t) x dt \to \frac{d{x}}{x} = -\frac{1}{2}\beta(t) dt \to \boldsymbol{\mu}_t = \mathbf{x}_0 e^{-\frac{1}{2}\int_0^t \beta(s)ds}$
	- sub-VP ODE: 同 VP
- ${\sigma}^2_t \,\mathbf{I}= var(\mathbf{x}_t)$ with i.c. ${\sigma}^2_0 = 0$  因爲 transition probability 在 $t=0$ 是 delta function
	- 方法: 解 forward SDE $dx=f dt + g dw$  
	- VE-SDE: $f= 0, g(t) = \sqrt{ \frac{ \mathrm{d}\left[\sigma^2(t)\right]}{\mathrm{d}t} }$.  注意這裏的 $\Sigma_t$ 加在 $\mathbf{x}_0$
		- 物理意義：$\mathbf{x}_t = \mathbf{x}_0 + {\sigma_t} \,\mathbf{z}_t = \mathbf{x}_0 + \sqrt{\sigma^2(t) - \sigma^2(0)} \,\mathbf{z}_t \approx \mathbf{x}_0 + \sigma(t)  \,\mathbf{z}_t$, because   ${\sigma}^2_t = \int_0^t g^2(s) ds \,\mathbf{I} = \int_0^t { \mathrm{d}\left[\sigma^2(s)\right]} \,\mathbf{I}= (\sigma^2(t) - \sigma^2(0))\mathbf{I}$ 
	- VP-SDE: $f=-\frac{1}{2}\beta(t)x$,  $d{x} = -\frac{1}{2} \beta(t) x dt + \sqrt{\beta(t)} dw$, 只能用 Fokker-Planck ?
	- ![[Pasted image 20250504001800.png]]
		- 物理意義：$\mathbf{x}_t = \mathbf{x}_0 e^{-\frac{1}{2} \int_0^t \beta(s) \mathrm{d} s}+ \sqrt{1-e^{-\int_0^t \beta(s) \mathrm{d} s}} \,\mathbf{z}_t = \sqrt{\alpha(t)} \mathbf{x}_0 + \sqrt{1-\alpha(t)} \,\mathbf{z}_t$.   
	- sub-VP-SDE:  $d{x} = -\frac{1}{2} \beta(t) x dt + \sqrt{\beta(t)\left( 1 - e^{-2\int_0^t \beta(s)\, \mathrm{d}s} \right)} dw$.  
		- 物理意義：$\mathbf{x}_t = \mathbf{x}_0 e^{-\frac{1}{2} \int_0^t \beta(s) \mathrm{d} s}+ \left[1-e^{-\int_0^t \beta(s) \mathrm{d} s}\right] \,\mathbf{z}_t = \sqrt{\alpha(t)} \mathbf{x}_0 + (1-\alpha(t)) \,\mathbf{z}_t$.   

一般定義 $\alpha(t) = e^{-\int_0^t \beta(s) \mathrm{d} s}$，可以看出 $\alpha(0)=1, \alpha(\infty)\to 0$, 這是一個遞減函數，和 $\beta(t)$ 相反。 
${\log\alpha(t)} = - \int_0^t \beta(s) ds \to  \frac{d \log\alpha(t)}{dt} = -\beta(t) \to -\frac{\dot{\alpha}(t)}{\alpha(t)} = \beta(t)$   如果 $\alpha(t)$ 遞減，導數的負值一定是正值。

我們可以再定義 $\bar{\beta}(t) = 1-\alpha(t) = 1- e^{-\int_0^t \beta(s) \mathrm{d} s}$,  很明顯 $\bar{\beta}(0)=0, \bar{\beta}(\infty)\to 1$.
$\bar{\beta}(t)$ 也是一個遞增函數。可以視爲一個 "normalized to 1" 的 $\beta(t)$, noise scheduling function.
- $t \to 0$:  $\bar{\beta}(t) = 1-\alpha(t) = 1- e^{-\int_0^t \beta(s) \mathrm{d} s} \approx \int_0^t \beta(s) \mathrm{d} s,\, \bar{\beta}(0)=0$
- $t\to \infty$: $\bar{\beta}(\infty) \to 1$


(**完全錯誤**)
VE:  signal 固定， noise 等加增加, SNR = S/N = S / kN 是 linear decrease. k = 1, 2, ...., log SNR = log S/N - log K
VP:  signal 等比下降，noise 等比增加,  SNR =  S*(b^k) / N/(b^k) = S/N * (b^k)^2 = S/N * b^(2k), b < 1, k = 1, 2, ...   => log SNR = log S/N + 2k * log b
subVP:  signal 等比下降，noise 等比增加 at lower rate, SNR =  S*(b^k) / N/(sqrt(b)^k) = S/N * (b^k)^1.5 = S/N * b^(1.5k), 
Based on Yang Song's paper.  VE: noise 等比數列,  VP: beta 是 等差數列。


VE, VP, sub-VP (all isotropic Gaussian) 結果整理如下：$N\left(\mathbf{x}_t; \boldsymbol{\mu}_t,{\sigma}^2_t\,\mathbf{I}\right)$  
或是 $\mathbf{x}_t = \boldsymbol{\mu}_t + \sqrt{\Sigma_t} \,\mathbf{z}_t$   where $\mathbf{z}_t \sim N(\mathbf{0}, \mathbf{I})$  and given $\mathbf{x}_0$
$$
p_{0 t}(\mathbf{x}_t \mid \mathbf{x}_0)= \begin{cases}N\left(\mathbf{x}_t ; \mathbf{x}_0,\left[\sigma^2(t)-\sigma^2(0)\right] \mathbf{I}\right), & \text {(VE)} \\ 
N\left(\mathbf{x}_t ; \mathbf{x}_0 e^{-\frac{1}{2} \int_0^t \beta(s) \mathrm{d} s},\left[1-e^{-\int_0^t \beta(s) \mathrm{d} s}\right] \mathbf{I}\right)  = N\left(\mathbf{x}_t ; \sqrt{\alpha(t)}\mathbf{x}_0 ,\left[1-\alpha(t)\right] \mathbf{I}\right)  & \text {(VP)}\\ 
N\left(\mathbf{x}_t ; \mathbf{x}_0 e^{-\frac{1}{2} \int_0^t \beta(s) \mathrm{d} s},\left[1-e^{-\int_0^t \beta(s) \mathrm{d} s}\right]^2 \mathbf{I}\right) = N\left(\mathbf{x}_t ; \sqrt{\alpha(t)}\mathbf{x}_0 ,\left[1-\alpha(t)\right]^2 \mathbf{I}\right)& \text {(sub-VP)}\end{cases}
$$
- VP vs. sub-VP:  mean 一樣，但是 variance (從 0 到 1) 因爲有平方增加比較慢。下圖比較三者的差別。$\sigma^2(t)$ 從 0.01 to 1 quadratically, $\beta(t)$ 從 0.1 到 20 linearly.
- 另外是一個 2D diffusion 圖。
![[Pasted image 20250504090729.png]]
![[Pasted image 20250504090754.png]]
#### Score Function of Continuous Transition Probability

雖是 continuous-time pdf, 其 score function, $\nabla_{\mathbf{x}_t}\log N(\boldsymbol{\mu}_t, {\sigma}^2_t\,\mathbf{I}) = -\frac{\mathbf{x}_t-\boldsymbol{\mu}_t}{ {\sigma}^2_t}$  還是 predict "scaled" additive noise!

再來計算: $\lambda(t) \propto 1 / \mathbb{E}\left[\left\|\nabla_{\mathbf{x}_t}\log N(\boldsymbol{\mu}_t, {\sigma}_t^2\,\mathbf{I})\right\|_2^2\right]$
- 分母 =  $\mathbb{E}\left[\|\frac{\mathbf{x}_t-\boldsymbol{\mu}_t}{ {\sigma}_t^2}\|^2\right] = \frac{ {\sigma}_t^2}{({\sigma}_t^2)^2} = {\sigma}_t^{-2}$ 
- $\lambda(t) \propto 1 / \mathbb{E}\left[\left\|\nabla_{\mathbf{x}_t}\log N(\boldsymbol{\mu}_t, {\sigma}_t^2\,\mathbf{I})\right\|_2^2\right] = {\sigma}_t^2$ : 是 transition probability 的 (Gaussian) variance!

#### Score Matching 物理意義

**代入 score function 和 $\lambda(t)$ 得到下式。說到底 Score matching 物理意義是 predict scheduled additive noise.** 和 NCSN 一樣

$$\begin{aligned}
\boldsymbol{\theta}^* &= \underset{\boldsymbol{\theta}}{\arg \min} \,\mathbb{E}_t \left\{ \sigma_t^2 \, \mathbb{E}_{\mathbf{x}_0} \, \mathbb{E}_{\mathbf{x}_t \mid \mathbf{x}_0} \left[ \left\| \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log N(\boldsymbol{\mu}_t, \Sigma_t) \right\|_2^2 \right] \right\}\\
&= \underset{\boldsymbol{\theta}}{\arg \min}  \,\mathbb{E}_t \left\{  \mathbb{E}_{\mathbf{x}_0} \, \mathbb{E}_{\mathbf{x}_t \mid \mathbf{x}_0} \left[ \sigma_t^{-2}\left\| \sigma_t^2\,\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) + \mathbf{x}_t -\boldsymbol{\mu}_t  \right\|_2^2 \right] \right\} \\
&= \underset{\boldsymbol{\theta}}{\arg \min}  \,\mathbb{E}_t \left\{  \mathbb{E}_{\mathbf{x}_0} \, \mathbb{E}_{\mathbf{x}_t \mid \mathbf{x}_0} \left[ \sigma_t^{-2}\left\| \sigma_t^2\,\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) + \sigma_t \,\mathbf{z}_t  \right\|_2^2 \right] \right\} \\
\end{aligned}
$$
其中 $\boldsymbol{\mu}_t\text{ and }\mathbf{x}_t - \boldsymbol{\mu}_t = {\sigma_t}\, \mathbf{z}_t$ ，由 (forward) diffusion SDE 決定。
Reminder: VE, VP, sub-VP 如下：
$$
p_{0 t}(\mathbf{x}_t \mid \mathbf{x}_0)= \begin{cases}N\left(\mathbf{x}_t ; \mathbf{x}_0,\left[\sigma^2(t)-\sigma^2(0)\right] \mathbf{I}\right), & \text {(VE)} \\ 
N\left(\mathbf{x}_t ; \mathbf{x}_0 e^{-\frac{1}{2} \int_0^t \beta(s) \mathrm{d} s},\left[1-e^{-\int_0^t \beta(s) \mathrm{d} s}\right] \mathbf{I}\right)  = N\left(\mathbf{x}_t ; \sqrt{\alpha(t)}\mathbf{x}_0 ,\left[1-\alpha(t)\right] \mathbf{I}\right)  & \text {(VP)}\\ 
N\left(\mathbf{x}_t ; \mathbf{x}_0 e^{-\frac{1}{2} \int_0^t \beta(s) \mathrm{d} s},\left[1-e^{-\int_0^t \beta(s) \mathrm{d} s}\right]^2 \mathbf{I}\right) = N\left(\mathbf{x}_t ; \sqrt{\alpha(t)}\mathbf{x}_0 ,\left[1-\alpha(t)\right]^2 \mathbf{I}\right)& \text {(sub-VP)}\end{cases}
$$

**接下來有兩種詮釋：1. noise prediction; 2. denoiser**

**VE-SDE:**  $\boldsymbol{\mu}_t = \mathbf{x}_0$  and ${\sigma}_t^2=\sigma^2(t)-\sigma^2(0)=\sigma^2(t)$ , $\mathbf{x}_t = \mathbf{x}_0 + \sigma(t) \,\mathbf{z}_t$ assuming $\sigma(0)= 0$. 

$$\begin{aligned}
\boldsymbol{\theta}^* &=  \underset{\boldsymbol{\theta}}{\arg \min}\,\mathbb{E}_t \left\{  \mathbb{E}_{\mathbf{x}_0} \, \mathbb{E}_{\mathbf{x}_t \mid \mathbf{x}_0} \left[ \sigma_t^{-2}\left\| \sigma_t^{2}\,\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) + \mathbf{x}_t -\boldsymbol{\mu}_t  \right\|_2^2 \right] \right\}\\
&= \underset{\boldsymbol{\theta}}{\arg \min}\,\mathbb{E}_t \left\{  \mathbb{E}_{\mathbf{x}_0} \, \mathbb{E}_{\mathbf{x}_t \mid \mathbf{x}_0} \left[ \sigma(t)^{-2}\left\| \sigma^2(t)\,\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) + \sigma(t) \mathbf{z}_t \right\|_2^2 \right] \right\}\\
&= \underset{\boldsymbol{\theta}}{\arg \min}\,\mathbb{E}_t \left\{  \mathbb{E}_{\mathbf{x}_0} \, \mathbb{E}_{\mathbf{x}_t \mid \mathbf{x}_0} \left[ \sigma(t)^{-2}\left\| \mathbf{D}_{\theta}(\mathbf{x}_t, t) -\mathbf{x}_0  \right\|_2^2 \right] \right\}\\
\end{aligned}
$$
1. Noise prediction: $\sigma^2(t) \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) \approx -\sigma(t) \mathbf{z}_t$
	1. Input $\mathbf{x}_t,\sigma^2(t)$; Output $\sigma(t) \mathbf{z}_t$
2. Denoiser: $\mathbf{D}_{\theta}(\mathbf{x}_t, t) = \mathbf{x}_t + \sigma^2(t)\,\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) \approx \mathbf{x}_0$
	1. Input $\mathbf{x}_t,\sigma^2(t)$; Output $\mathbf{x}_0$


**VP-SDE:**  $\boldsymbol{\mu}_t = \sqrt{\alpha(t)}\mathbf{x}_0$  and ${\sigma}_t^2=1-\alpha(t)$ , $\mathbf{x}_t = \sqrt{\alpha(t)}\mathbf{x}_0 + \sqrt{1-\alpha(t)} \,\mathbf{z}_t$

$$\begin{aligned}
\boldsymbol{\theta}^* &=  \underset{\boldsymbol{\theta}}{\arg \min}\,\mathbb{E}_t \left\{  \mathbb{E}_{\mathbf{x}_0} \, \mathbb{E}_{\mathbf{x}_t \mid \mathbf{x}_0} \left[\sigma_t^{-2} \left\| \sigma_t^2\,\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) + \mathbf{x}_t -\boldsymbol{\mu}_t  \right\|_2^2 \right] \right\}\\
&=  \underset{\boldsymbol{\theta}}{\arg \min}\,\mathbb{E}_t \left\{  \mathbb{E}_{\mathbf{x}_0} \, \mathbb{E}_{\mathbf{x}_t \mid \mathbf{x}_0} \left[ (1-\alpha(t))^{-1}\left\| (1-\alpha(t))\,\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) + \mathbf{x}_t -\sqrt{\alpha(t)}\mathbf{x}_0  \right\|_2^2 \right] \right\}\\
&= \underset{\boldsymbol{\theta}}{\arg \min}\,\mathbb{E}_t \left\{  \mathbb{E}_{\mathbf{x}_0} \, \mathbb{E}_{\mathbf{x}_t \mid \mathbf{x}_0} \left[ (1-\alpha(t))^{-1}\left\| (1-\alpha(t))\,\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) + \sqrt{1-\alpha(t) } \mathbf{z}_t  \right\|_2^2 \right] \right\}\\
&= \underset{\boldsymbol{\theta}}{\arg \min}\,\mathbb{E}_t\left\{  \mathbb{E}_{\mathbf{x}_0} \, \mathbb{E}_{\mathbf{x}_t \mid \mathbf{x}_0} \left[ (1-\alpha(t))^{-1}\left\| \mathbf{D}_{\theta}(\mathbf{x}_t, t) -\sqrt{\alpha(t)}\mathbf{x}_0  \right\|_2^2 \right] \right\}\\
\end{aligned}
$$

1. Noise prediction: $(1-\alpha(t)) \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) \approx -\sqrt{1-\alpha(t)} \mathbf{z}_t$
	1. Input $\mathbf{x}_t,\alpha(t)$; Output $\sqrt{1-\alpha(t)} \mathbf{z}_t$
2. Denoiser: $\mathbf{D}_{\theta}(\mathbf{x}_t, t) = \mathbf{x}_t + (1-\alpha(t))\,\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) \approx \sqrt{\alpha(t)}\mathbf{x}_0$
	1. Input $\mathbf{x}_t,1-\alpha(t)$; Output $\sqrt{\alpha(t)} \mathbf{x}_0$, scaled original images


**sub-VP-SDE:**  $\boldsymbol{\mu}_t = \sqrt{\alpha(t)}\mathbf{x}_0$  and ${\sigma}_t^2=(1-\alpha(t))^2$ , $\mathbf{x}_t = \sqrt{\alpha(t)}\mathbf{x}_0 + (1-\alpha(t)) \,\mathbf{z}_t$

$$\begin{aligned}
\boldsymbol{\theta}^* &=  \underset{\boldsymbol{\theta}}{\arg \min}\,\mathbb{E}_t \left\{  \mathbb{E}_{\mathbf{x}_0} \, \mathbb{E}_{\mathbf{x}_t \mid \mathbf{x}_0} \left[ \left\| \Sigma_t\,\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) + \mathbf{x}_t -\boldsymbol{\mu}_t  \right\|_2^2 \right] \right\}\\
&=  \underset{\boldsymbol{\theta}}{\arg \min}\,\mathbb{E}_t \left\{  \mathbb{E}_{\mathbf{x}_0} \, \mathbb{E}_{\mathbf{x}_t \mid \mathbf{x}_0} \left[ \left\| (1-\alpha(t))^2\,\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) + \mathbf{x}_t -\sqrt{\alpha(t)}\mathbf{x}_0  \right\|_2^2 \right] \right\}\\
&= \underset{\boldsymbol{\theta}}{\arg \min}\,\mathbb{E}_t \left\{  \mathbb{E}_{\mathbf{x}_0} \, \mathbb{E}_{\mathbf{x}_t \mid \mathbf{x}_0} \left[ \left\| (1-\alpha(t))^2\,\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) + (1-\alpha(t)) \mathbf{z}_t  \right\|_2^2 \right] \right\}\\
&= \underset{\boldsymbol{\theta}}{\arg \min}\,\mathbb{E}_t\left\{  \mathbb{E}_{\mathbf{x}_0} \, \mathbb{E}_{\mathbf{x}_t \mid \mathbf{x}_0} \left[ \left\| \mathbf{D}_{\theta}(\mathbf{x}_t, t) -\sqrt{\alpha(t)}\mathbf{x}_0  \right\|_2^2 \right] \right\}\\
\end{aligned}
$$

1. Noise prediction: $(1-\alpha(t))^2 \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) \approx -(1-\alpha(t)) \mathbf{z}_t$
	1. Input $\mathbf{x}_t,\alpha(t)$; Output $(1-\alpha(t)) \mathbf{z}_t$
2. Denoiser: $\mathbf{D}_{\theta}(\mathbf{x}_t, t) = \mathbf{x}_t + (1-\alpha(t))^2\,\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) \approx \sqrt{\alpha(t)}\mathbf{x}_0$
	1. Input $\mathbf{x}_t,1-\alpha(t)$; Output $\sqrt{\alpha(t)} \mathbf{x}_0$, scaled original images


#### Continuous-time SDE, 定義 VE (variance explode, 類似 NCSN 的 continuous version), 和 VP (variance preserve, 類似 DDPM), 以及作者建議的 sub-VP methods.

### VE (explode) forward continuous SDE:
VE 就是 NCSN 的 continuous version,  所以我們可以從 discrete version 開始:

Forward path (training): $\tilde{\mathbf{x}} = \mathbf{x} + \sigma_i \mathbf{z}$.  使用 $\sigma_{min} = \sigma_1 < \cdots < \sigma_N = \sigma_{max}$

我們可以改寫成：$\mathbf{x}_i = \mathbf{x}_0 + \sigma_i \mathbf{z}_i$.  where $\mathbf{x}_0$ 是 clean image.  $0= \sigma_0 < \sigma_1 < \cdots < \sigma_N = \sigma_{max}$
$\mathbf{z}_i \sim N(0, I)$ 是 white Gaussian noise.  理論上 $\sigma_{max}$ 可以是無限大，因此稱為 VE - variance explode.

VE Forward path 可以改寫成 recursive  (Markov chain) 形式：
$$
\mathbf{x}_i = \mathbf{x}_{i-1} + \sqrt{\sigma_i^2 - \sigma_{i-1}^2} \, \mathbf{z}_{i-1}, \quad i=1,2,\cdots,N
$$
上式可以驗證如下
$$\begin{aligned}
\mathbf{x}_i &= \mathbf{x}_{i-1} + \sqrt{\sigma_i^2 - \sigma_{i-1}^2} \, \mathbf{z}_{i-1} \\ 
&= \mathbf{x}_{i-2} + \sqrt{\sigma_i^2 - \sigma_{i-1}^2} \, \mathbf{z}_{i-1} + \sqrt{\sigma_{i-1}^2 - \sigma_{i-2}^2} \, \mathbf{z}_{i-2} \\
&= \mathbf{x}_{i-2} + \sqrt{\sigma_i^2 - \sigma_{i-2}^2} \, \mathbf{z}'_{i-2} \\
&= \mathbf{x}_{0} + \sqrt{\sigma_i^2 - \sigma_{0}^2} \, \mathbf{z}'_{0} =\mathbf{x}_{0} + \sigma_i  \, \mathbf{z}'_{0} \\
\end{aligned}$$

接下來把 discrete recursive 形式變成無窮小 time step 得到 continuous SDE forward path: 

$$
\mathrm{d}\mathbf{x}_t = \sqrt{ \frac{ \mathrm{d}\left[\sigma^2_t\right]}{\mathrm{d}t} } \, \mathrm{d}\mathbf{w}_t
$$
注意這裏的 $\sigma_t$ 是加在原始 image $\mathbf{x}_0$:    $\mathbf{x}_t = \mathbf{x}_0 + \sigma_t \,\mathbf{z}_t$.
其中 $\sigma_t\ge 0$ 而且是**遞增函數！稱為 noise scheduling function.**  我們可以假設 $\sigma(0)=0$ 對應原始無 noise images $\mathbf{x}_0$.

- $f(\mathbf{x}_t, t) = 0$； $g(t) = \sqrt{ \frac{ \mathrm{d}\left[\sigma^2(t)\right]}{\mathrm{d}t} }$；or $g^2(t) = 2 \sigma(t) \dot{\sigma}(t)$ 

VE reverse SDE (sampling): $d \mathbf{x}_t = - g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})dt+ g(t) \,d\mathbf{w}_t = - 2 \sigma(t) \dot{\sigma(t)}\nabla_{\mathbf{x}} \log p_t(\mathbf{x})dt+ \sqrt{2 \sigma(t) \dot{\sigma(t)}} \,d\mathbf{w}_t$

VE ODE 看起來非常簡潔： $d \mathbf{x}_t =  - \frac{1}{2} g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})dt = - \sigma(t) \dot{\sigma(t)}\nabla_{\mathbf{x}} \log p_t(\mathbf{x})dt \approx - \sigma(t) \dot{\sigma(t)}\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)dt$

利用 score matching 得到的 $\mathbf{s}(\mathbf{x}_t, t)$ 做 sampling, like DDIM?


### VP (preserve) forward SDE:
VP 就是 DDPM 的 continuous version,  所以我們可以從 discrete version 開始。

VP Forward path 本身就已經是 Markov chain 的形式如下。
因為 signal^2 + noise^2 = constant,  所以稱為 variance preserve SDE!

$$
\mathbf{x}_i=\sqrt{1-\beta_i} \mathbf{x}_{i-1}+\sqrt{\beta_i} \mathbf{z}_{i-1}, \quad i=1, \cdots, N .
$$

把 dt 變成無窮小，利用二次公司展開 $(1-\beta_i)^{1/2} \approx 1 - \frac{1}{2} \beta(t)$。所以得到以下的 forward SDE.
$$
\mathrm{d} \mathbf{x}_t = -\frac{1}{2} \beta(t) \mathbf{x}_t \mathrm{d} t+\sqrt{\beta(t)} \mathrm{d} \mathbf{w}_t
$$
其中 $\beta(t)\ge 0$ 而且是**遞增函數！也是 noise scheduling function.**  我們可以假設 $\beta(0)=0$ 對應原始無 noise images $\mathbf{x}_0$.

(Correct explanation) $\mathbf{x}_t = \mathbf{x}_0 e^{-\frac{1}{2} \int_0^t \beta(s) \mathrm{d} s}+ \sqrt{1-e^{-\int_0^t \beta(s) \mathrm{d} s}} \,\mathbf{z}_t = \sqrt{\alpha(t)} \mathbf{x}_0 + \sqrt{1-\alpha(t)} \,\mathbf{z}_t$.  如果 $\mathbf{x}_0$ 的 variance 為 1, 整個 forward path 的 variance 固定為 1, 因此稱爲 variance preserve.

(Another explanation using OU process above) 如果 $S = D/\theta=\beta(t)/\beta(t)=1$ , variance preserve.  只要 initial 是 standard normal distribution with variance = 1，而且 $g^2(t)/\theta(t)=1$,  variance preserve.

- $f(\mathbf{x}_t, t) = -\frac{1}{2}\beta(t)\mathbf{x}_t$； $g(t) = \sqrt{ \beta(t)}$；or $g^2(t) = \beta(t)$

VP Backward SDE for sampling: $d \mathbf{x}_t = -\frac{1}{2}\beta(t)\mathbf{x}_t d t- \beta(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t)dt+ \sqrt{\beta(t)} \,d\mathbf{w}_t$
VP ODE: $d \mathbf{x}_t = -\frac{1}{2}\beta(t)\mathbf{x}_t d t- \frac{1}{2} \beta(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t)dt = -\frac{1}{2}\beta(t)[\mathbf{x}_t+\nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t)]dt$

#### Sub-VP (preserve) forward continuous SDE:

Discrete version forward path 是 Markov chain?  NO!
Variance preserve?  maybe!
$$
\mathbf{x}_{i-1} = \frac{1}{\sqrt{1-\beta_i}} \left( \mathbf{x}_i + \beta_i \, \mathbf{s}_{\boldsymbol{\theta}^*}(\mathbf{x}_i, i) \right) + \sqrt{\beta_i} \, \mathbf{z}_i, \quad i = N, N-1, \cdots, 1
$$

看不出物理意義？

$$
\mathrm{d}\mathbf{x} = -\frac{1}{2} \beta(t) \mathbf{x} \, \mathrm{d}t + \sqrt{ \beta(t) \left( 1 - e^{-2\int_0^t \beta(s)\, \mathrm{d}s} \right) } \, \mathrm{d}\mathbf{w}
$$

the following is redundant?
$$
\mathbf{x}_i^m = \mathbf{x}_i^{m-1} + \epsilon_i \, \mathbf{s}_{\boldsymbol{\theta}^*}(\mathbf{x}_i^{m-1}, \sigma_i) + \sqrt{2 \epsilon_i} \, \mathbf{z}_i^m, \quad m=1,2,\cdots, M
$$

Flow match : Diffusion,   OT (Optimal Transport)

## SDE Sampling

**Sampling: 這是 continuous SDE 和 discrete NCSN or DDPM 主要的差異。**

**已經有很多的 SDE solver 可以直接利用。但是慢，兩種解法：**

1. **Prediction - Correction**
2. **使用 ODE 取代 SDE**

![[Pasted image 20250504004724.png]]


## SDE (samples) to ODE (probability flow) (DDPM to DDIM?)

Lagenvin SDE 提供的是 samples,  就是像布朗運動的 samples.   但是速度很慢。如果要加速，一個方法是轉換成 ODE (Ordinary Differential Equation),  就有種種不同的加速工具。

![[Pasted image 20250202115412.png]]


## Diffusion 加速 Distillation:  ODE to CTM (long jump)
Teacher model: ODE
Student model: long jump, CTM

![[Pasted image 20250202151459.png]]



## Conditional Diffusion and Inverse Problem

一般應用是 given conditions: (1) class (如 dog, cat), (2) text input,  (3) image input.
可以把 condition 展開成如下的方法。$\mathbf{x}$ 是 image, y 是 control signal, e.g. class, or te\mathbf{x}t and image?

![[Pasted image 20250202123612.png]]



## Unified View for Discrete and Continuous Diffusion?

![[Pasted image 20250507225449.png]]

Training, OK
$$
\mathcal{L}(\theta)=\mathbb{E}_{\sigma, \boldsymbol{X}, \boldsymbol{N}}\left[w(\sigma)\left\|\boldsymbol{X}-D_\theta(\boldsymbol{X}+\sigma \boldsymbol{N}, \sigma)\right\|_2^2\right], \quad \boldsymbol{X} \sim f_{\boldsymbol{X}}
$$
Sampling, NOK?
$$
\begin{aligned}
\boldsymbol{x}_{k+1} & =\boldsymbol{x}_k+\frac{\tau_k}{\sigma_k^2}\left(D_\theta\left(\boldsymbol{x}_k, \sigma_k\right)-\boldsymbol{x}_k\right)+\sqrt{2 \tau_k \mathcal{T}_k} \boldsymbol{N} \\
& =\boldsymbol{x}_k+\tau_k \nabla \log f_{\boldsymbol{X}_{\sigma_k}}\left(\boldsymbol{x}_k\right)+\sqrt{2 \tau_k \mathcal{T}_k} \boldsymbol{N}
\end{aligned}
$$

重點是有 forward, 就可以得到 backward.  帶入，reverse SDE 和 forward SDE 一樣，只是 $dt \to -dt$ 或是走向負梯度的方向。也就是往高機率的地方前進。
- Minimal 值 (如 loss function) 是往負梯度方向走。GD:   $\theta_{n+1}=\theta_n − \alpha \nabla L(\theta_n)$
- Maximal 值 (如 likelihood) 是往梯度方向走。



## Reference

Yang Song, PPDM, ICLR 2021: https://www.youtube.com/watch?v=L9ZegT87QK8&ab_channel=ArtificialIntelligence

SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS
https://arxiv.org/pdf/2011.13456

https://www.youtube.com/watch?v=ud6z5SkjoZI&t=2098s&ab_channel=BainCapitalVentures



## Appendix

我們先保留 $g(t)$,  再用 Fokker Planck 看如何和 $\sigma$ 對應。

**VE**
- $f(\mathbf{x}_t, t) = 0$； keep $g(t)$
- Forward: $d {\mathbf{x}} = g(t) \,d\mathbf{w}$
- Backward: $d {\mathbf{x}} = - g(t)^2 \, \nabla_\mathbf{x} \log p_t(\mathbf{x}) dt +  g(t)\,d\mathbf{w}$
- ODE: $d {\mathbf{x}} = -  \frac{1}{2} g(t)^2\,\nabla_\mathbf{x} \log p_t(\mathbf{x}) dt$

從 forward 可以看到 g(t) 基本就是 noise scheduling.  如何 related to ?


Background: Noise Schedule & Marginal Variance

In score-based diffusion models (like DDPM or VE/VP SDEs), the **marginal distribution** at time $t$ is modeled as:

$$
\mathbf{x}_t \sim N(0, \sigma^2(t) I)
$$

That is, the **variance of the sample** at time $t$ is given by $\sigma^2(t)$, and $\Sigma_t$ defines the **noise schedule**.

So if we define the variance of the noise as:

$$
\sigma^2(t) = \text{Var}[\mathbf{x}_t] = \int_0^t g^2(s)\, ds
$$

Then by **differentiating both sides**:

$$
\frac{d}{dt} \sigma^2(t) = g^2(t)
$$

Also note:

$$
\frac{d}{dt} \sigma^2(t) = \frac{d}{dt} \left(\sigma_t^2\right) = 2\sigma_t\dot{\sigma}(t)
$$

So we equate:

$$
2\sigma_t \dot{\sigma}(t) = g^2(t)
$$

Thus:

$$
\boxed{
\frac{1}{2} g^2(t) = \sigma_t \dot{\sigma}(t)
}
$$

Now, if we plug that into the probability flow ODE, we get:

$$
\frac{d\mathbf{x}}{dt} = -\sigma_t \dot{\sigma}(t) \nabla \log p(\mathbf{x}, t)
$$

Which matches the form:

$$
\frac{d\mathbf{x}}{dt} = -\dot{\sigma}(t) \sigma_t \nabla \log p(\mathbf{x}; \Sigma_t)
$$



#### OU Process
![[Pasted image 20250430222922.png]]
P(x, t=0 | x‘, t=0) = N(x', 0), var = 0,    P(x, t |x', t=0) = N(ll,  D/theta (1-e...))
if $\theta \to 0, D/\theta * (2\theta(t-t'))=D*2*(t-t')=\sigma^2 (t-t')$ ->  $N(x', \sigma^2 (t-t'))$




