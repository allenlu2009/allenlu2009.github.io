---
title: Math AI - Stochastic Differential Equation Forward 
date: 2023-05-01 23:10:08
categories:
- Math_AI
tags: [SDE, Stochastic equations, diffusion equations, reverse-time equations, Fokker-Planck equations]
typora-root-url: ../../allenlu2009.github.io
---


## Reference

[PII: 0304-4149(82)90051-5 (core.ac.uk)](https://core.ac.uk/download/pdf/82826666.pdf)  @andersonReversetimeDiffusion1982 : 專門針對 reverse time SDE equations. 

http://pordlabs.ucsd.edu/pcessi/theory2019/gardiner_ito_calculus.pdf

Ito equations



## Introduction





## 隨機過程簡單系統

**一個問題是隨機過程簡單系統有什麽實際用途？**

非常多，例如 noisy control system, robust control  system, finance system



**更進一步，隨機過程的時光倒流 (減熵) 有什麽用途？**

信號處理的 smoothing 問題 

最近大熱門的 diffusion process 就是利用時光倒流來 train 一個 neural network.

之後可以用 noise 產生 image.  更廣汎說，可以從一個機率分佈 transform 到另一個機率分佈。作爲生成式 AI 或是 style transfer AI.



**有幾類隨機過程問題：**

* Signal + noise estimate the original signal (Kalman filter, Kalman-Bucy filter)
* 布朗運動 : 描述現象 only?
* Distribution A + noise : diffusion denoise



**此處我們討論隨機微分方程 (Ito equation) 的 forward-time 和 backward-time 的形式和對應 Fokker-Planck equations。**



### (Forward) Ito Differential Equation

我們從 general 的 stochastic differential equation (SDE) 開始：
$$
\begin{align}
d x_t=f\left(x_t, t\right) d t+g\left(x_t, t\right) d w_t
\end{align}
$$
where $w_t$ has the usual properties. (就是 Wiener process, white noise)



以上是 random samples or process 的微分方程。可以用來作爲 Monte Carlo 模擬，但無法直接計算機率分佈如何隨時間變化。所以可以導入機率分佈的偏微分方程 Fokker-Planck equation 如下。



#### Forward unconditioned equation (Fokker-Planck equation)

也就是 Fokker-Planck Equations, yields
$$
\begin{align}
-\frac{\partial p\left(x_t, t\right)}{\partial t}= & \sum_i \frac{\partial}{\partial x_t^i}\left[p\left(x_t, t\right) f^i\left(x_t, t\right)\right] 
 -\frac{1}{2} \sum_{i, j, k} \frac{\partial^2\left[g^{i k}\left(x_t, t\right) g^{j k}\left(x_t, t\right) p\left(x_t, t\right)\right]}{\partial x_t^i \partial x_t^j} .
\end{align}
$$
以上可以説是 general diffusion equation.

* 左邊是機率分佈對時間微分。

* 右邊第二項是機率分佈對空間的二次微分，就是機率分佈 Laplacian (空間 tension)。
* 右邊第一項是機率分佈的 transient 項？如果 eigenvalue 實部小於 0, 會隨時間 exponential 消失。





## Ito Differential Equation Examples



### Coefficients Without $x$ Dependence (Forward)

$\mathrm{d}x = a(t) \,\mathrm{d}t + b(t)\, \mathrm{d}W(t)$



此處 $a(t)$ 和 $b(t)$ 是 non-random function of time.  上式兩側積分：

$x(t) = x_0 + \int_{t=t_0}^{t} a(t) \,\mathrm{d}t + \int_{t=t_0}^{t} b(t)\, \mathrm{d}W(t)$



因爲 $W(t)$ 是 Wiener (zero-mean Gaussian) process.  $x(t)$ 可以視爲 Gaussian random variable 的 linear combination, 還是 zero-mean Gaussian.    

這個 Gaussian process 的 mean:


$$
\langle x(t)\rangle=\left\langle x_0\right\rangle+\int_{t_0}^t a(t) d t
$$
因爲右邊第二項 Ito 積分的 mean 為 0.  再來計算 auto-correlation (including variance):
$$
\begin{align}
\langle[x(t) & -\langle x(t)\rangle][x(s)-\langle x(s)\rangle]\rangle \equiv\langle x(t), x(s)\rangle \\
& =\left\langle\int_{t_0}^t b\left(t^{\prime}\right) d W\left(t^{\prime}\right) \int_{t_0}^s b\left(s^{\prime}\right) d W\left(s^{\prime}\right)\right\rangle=\int_{t_0}^{\min (t, s)}\left[b\left(t^{\prime}\right)\right]^2 d t^{\prime}
\end{align}
$$
有了 Gaussin 的 mean and variance 就可以完全決定 $x(t)$.

* Random process 的 mean depends on $a(t)$ 對時間的積分。
* Random process 的 variance 隨著時間增加 ( $b(t)^2 > 0$ )。不過是否收斂 depending on $b(t)^2$.  如果是負的 exponential function 則收斂。
* 非 stationary process!
* $x(t)$ 的 frequency domain spectrum 就是 FFT of auto-correlation function.



### Multiplicative Linear White Noise Process (Forward)

$\mathrm{d}x = c x\, \mathrm{d}W(t)$

其 sample 的解形式如下 (make y = log x) :
$$
x(t)=x\left(t_0\right) \exp \left\{c\left[W(t)-W\left(t_0\right)\right]-\frac{1}{2} c^2\left(t-t_0\right)\right\}
$$
We can calculate the mean by using the formula for any Gaussian variable $z$ with zero mean
$$
\langle\exp z\rangle=\exp \left(\left\langle z^2\right\rangle / 2\right)
$$
so that
$$
\begin{aligned}
\langle x(t)\rangle=\left\langle x\left(t_0\right)\right\rangle & \exp \left[\frac{1}{2} c^2\left(t-t_0\right)-\frac{1}{2} c^2\left(t-t_0\right)\right] \\
= & \left\langle x\left(t_0\right)\right\rangle .
\end{aligned}
$$
This result is also obvious from definition, since
$$
\begin{align}
& \langle \mathrm{d} x\rangle=\langle c x \, \mathrm{d} W(t)\rangle=0 \quad \text { so that } \\
& \frac{\mathrm{d}\langle x\rangle}{\mathrm{d} t}=0 .
\end{align}
$$
We can also calculate the autocorrelation function
$$
\begin{aligned}
\langle x(t) x(s)\rangle & =\left\langle x\left(t_0\right)^2\right\rangle\left\langle\exp \left\{c\left[W(t)+W(s)-2 W\left(t_0\right)\right]-\frac{1}{2} c^2\left(t+s-2 t_0\right)\right\}\right\rangle \\
& =\left\langle x\left(t_0\right)^2\right\rangle \exp \left\{\frac{1}{2} c^2\left[\left\langle\left[W(t)+W(s)-2 W\left(t_0\right)\right]^2\right\rangle-\left(t+s-2 t_0\right)\right]\right\} \\
& =\left\langle x\left(t_0\right)^2\right\rangle \exp \left\{\frac{1}{2} c^2\left[t+s-2 t_0+2 \min (t, s)-\left(t+s-2 t_0\right)\right]\right\} \\
& =\left\langle x\left(t_0\right)^2\right\rangle \exp \left[c^2 \min \left(t-t_0, s-t_0\right)\right]
\end{aligned}
$$

* Random process 的 mean 不隨時間變化。Multiplicative white noise.
* Random process 的 variance depends on $c > 1$ or $c <1$.  如果大於 1, variance exponentially 增加。
* 非 stationary process! 
* $x(t)$ 的 frequency domain spectrum 就是 FFT of auto-correlation function.



### Ornstein-Uhlenbeck Process (Forward)

假設 $x$ 是 non-deterministic $n$-dimentional process, 並由下述綫性微分方程表示：

$$
\begin{align}
\mathrm{d} x=A x \mathrm{~d} t+B \mathrm{~d} w . \\
\end{align}
$$

其中 $A, B$ 都是常數矩陣。$w$ 是 vector Wiener process.  
$$
\begin{align}
x(t)= x(0) \mathrm{e}^{At} + \int_{0}^t \mathrm{e}^{A(t-s)} B \mathrm{~d} w(s) 
\end{align}
$$

$$
\langle x(t)\rangle=\left\langle x_0\right\rangle \mathrm{e}^{At}
$$

The correlation function follows similarly
$$
\begin{aligned}
\left\langle\boldsymbol{x}(t), \boldsymbol{x}^{\mathrm{T}}(s)\right\rangle \equiv & \left\langle[\boldsymbol{x}(t)-\langle\boldsymbol{x}(t)\rangle][\boldsymbol{x}(s)-\langle\boldsymbol{x}(s)\rangle]^{\mathrm{T}}\right\rangle \\
= & \mathrm{e}^{A t}\left\langle\boldsymbol{x}(0), \boldsymbol{x}^{\mathrm{T}}(0)\right\rangle \mathrm{e}^{A^T s}  +\int_0^{\min (t, s)} \mathrm{e}^{A\left(t-t^{\prime}\right)} B B^{\mathrm{T}} \mathrm{e}^{A^T\left(s-t^{\prime}\right)} d t'
\end{aligned}
$$

 

#### (Variance) Stationary Version ($A$ 的 eigenvalues 實部小於 0)

如果 $A$ 的 eigvalues 實部小於 0，存在 wide-sense stationary solution (mean and variance):
$$
\begin{align}x(t)=\int_{-\infty}^t \mathrm{e}^{A(t-s)} B \mathrm{~d} w(s) .\end{align}
$$

$$
\langle x(t)\rangle= 0
$$

$$
\begin{aligned}\left\langle\boldsymbol{x}(t), \boldsymbol{x}^{\mathrm{T}}(s)\right\rangle = \int_{-\infty}^{\min (t, s)} \mathrm{e}^{A\left(t-t^{\prime}\right)} B B^{\mathrm{T}} \mathrm{e}^{A^T\left(s-t^{\prime}\right)} d t'\end{aligned}
$$

上式乍看之下很難是 stationary，但是如果我們定義 (time average, 不是 space average!)
$$
\begin{aligned}\sigma = \left\langle\boldsymbol{x}(t), \boldsymbol{x}^{\mathrm{T}}(t)\right\rangle \end{aligned}
$$
可以證明  $A \sigma + \sigma A^T = B B^T$  [reference Gardiner Ito Calculus]

因爲 $A, B$ 都是 constant matrix, 所以 $\sigma$ 本身也是 constant, 也就是 stationary!
$$
\begin{align}P=E\left[x(t) x^{T}(t)\right] = \left<x(t)x^T(t)\right> = \sigma .\end{align}
$$







## Appendix

