---
title: Math AI - Stochastic Differential Equation Backward 
date: 2023-04-29 23:10:08
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

時光倒流 (time reversal) 在日常生活幾乎不可能。所有的觀察 (非實驗結果) 歸納出熱力學第二定律：熵隨時間增加。因此複雜系統的演進過程，例如生米煮成熟飯，只能一個方向發展，時光倒流看起來不可能發生。但在微觀系統的方程式並沒有時光倒流的矛盾s問題，也就是 forward-time 和 backward-time 都是完全合理物理運動，而且對稱。只要把方程式的 $t \to -t$ 基本就可以描述 time reversal.

回到日常生活，對於 (非隨機過程) 簡單系統，例如撞球或是星球運行，時光倒流是合理可以發生。同樣也是把方程式的 $t \to -t$ 就可以描述。



**但對於隨機過程簡單系統，這就非常有趣：到底是類似熱力學熵增加，無法時光倒流。還是比較類似微觀或簡單系統存在時光倒流，並可用簡單的 time symmetry, 也就是 $t \to -t$, 就可以描述?** 

**答案是：Yes, 存在時光倒流。No, 不是簡單的 time symmetry ($t \to -t$).**



其他問題：

1. 簡單隨機系統有熵 (entropy) 增加的概念？或甚至 governed by entropy increase law?
   * 因爲有 randomness, 所以還是有 entropy 概念？**但不一定是熵增加**。**在 non-stationary transient process, 還是可以做時光倒流 (time reversal) 或是熵減小**。

  

|                | Forward-time                                       | Backward-time        |
| -------------- | -------------------------------------------------- | -------------------- |
| 複雜系統       | 熱力學**熵增加**                                   | No (熵無法變小)      |
| 微觀系統       | 量子力學，場論 (no entropy)                        | Yes, $t \to -t$      |
| 簡單非隨機系統 | 古典力學 (no entropy)                              | Yes, $t \to -t$      |
| 簡單隨機系統   | 隨機微分方程 (SDE): Ito or Fokker-Planck equations | **Yes, reverse SDE** |

<img src="https://upload.wikimedia.org/wikipedia/commons/f/f2/FokkerPlanck.gif" alt="img" style="zoom:100%;" />

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



#### Backward (not reverse-time, not $t \to -t$) Kolmogorov equation 

for $s \geqslant t$ is

$$
\begin{align}
-\frac{\partial p\left(x_s, s \mid x_t, t\right)}{\partial t}= & \sum_i f^i\left(x_t, t\right) \frac{\partial p\left(x_s, s \mid x_t, t\right)}{\partial x_t^i}  +\frac{1}{2} \sum_{i, j, k} g^{i k}\left(x_t, t\right) g^{j k}\left(x_t, t\right) \frac{\partial^2 p\left(x_s, s \mid x_t, t\right)}{\partial x_t^i \partial x_t^j}
\end{align}
$$



### (Backward) Ito Differential Equation

推導參考 reference and appendix.  Backward Ito Differential Equation 如下：

$$
\begin{align}
\mathrm{d} \bar{x}_t=\bar{f}\left(\bar{x}_t, t\right) \mathrm{d} t+g\left(\bar{x}_t, t\right) \mathrm{d} \bar{w}_t
\end{align}
$$


where $\bar{f}^i$ is:
$$
\begin{align}
\bar{f}^i\left(x_t, t\right)=f^i\left(x_t, t\right)-\frac{1}{p\left(x_t, t\right)} \sum_{i, k} \frac{\partial}{\partial x_t^i}\left[p\left(x_t, t\right) g^{i k}\left(x_t, t\right) g^{i k}\left(x_t, t\right)\right]
\end{align}
$$







## Ito Differential Equation Examples




### 綫性 Ornstein-Uhlenbeck Equation (Foward and Backward)

假設 $x$ 是 non-deterministic, wide-sense stationary $n$-dimentional process, 並由下述綫性微分方程表示：

$$
\begin{align}
\mathrm{d} x=A x \mathrm{~d} t+B \mathrm{~d} w . \\
\end{align}
$$

其中 $A, B$ 都是常數矩陣。$A$ 的 eigvalues 實部小於 0.   $w$ 是 vector Wiener process.  

這個 forward-time 的解形式 (representation) 如下：
$$
\begin{align}
x(t)=\int_{-\infty}^t \mathrm{e}^{A(t-s)} B \mathrm{~d} w(s) .
\end{align}
$$

此處假設 $A$ 的 eigenvalues 實部小於 0 (隨時間增加收斂)。



我們可能會推測 reverse-time 的解形式 (representation) 如下。但這是 time-reverse ($t\to -t$) 的想法。就是 $x(t)$ 時間減小而變大。
$$
\begin{align}
x(t)=-\int_t^{\infty} \mathrm{e}^{\bar{A}(t-s)} \bar{B} \mathrm{~d} \bar{w}(s) . \\
\end{align}
$$



我們這裏想得到的是一個 **forward-time representation** of "$x(t)$ a reverse-time representation".   就是把 reverse-time 轉換 (unfold) 成 forward-time 的 solution.  此時新的 $x(t)$ 隨著時間增加而變大。所以我們要魔改原始的 Ito SDE。A reverse-time SDE is：
$$
\begin{align}
\mathrm{d} x=\bar{A} x \mathrm{~d} t+\bar{B} \mathrm{~d} \bar{w} \\
\end{align}
$$

此處 $\bar{A}$ 的 eigenvalues 實部大於 0.  
$$
\begin{align}
P=E\left[x(t) x^{\prime}(t)\right] .
\end{align}
$$

The matrix $P$ is the solution of the linear matrix equation $P A^{\prime}+A P=-B B^{\prime}$, and is nonsingular precisely when rank $\left[B ,A B, \cdots A^{n-1} B\right]=n$.

Suppose $P$ is nonsingular, and define a vector process $\bar{w}$ by
$$
\begin{align}
\mathrm{d} \bar{w}=\mathrm{d} w-B^{\prime} P^{-1} x \mathrm{~d} t, \quad \bar{w}(0)=0,
\end{align}
$$
which in conjunction with 原始的 Ito equation implies
$$
\begin{align}
\mathrm{d} x=\left(A+B B^{\prime} P^{-1}\right) \mathrm{d} t+B \mathrm{~d} \bar{w} = \bar{A} \mathrm{d}t + \bar{B} \mathrm{d}\bar{w}
\end{align}
$$

where 

$$
\begin{align}
\bar{A} = A + B B' P^{-1}; \quad \bar{B} = B
\end{align}
$$

可以證明 $\bar{A} = A + B B' P^{-1}$ 的 eigenvalues 實部大於 0。 $\bar{w}(t)$ 是 vector Wiener process with $x(t)$ independent of past increments of $\bar{w}$, but not of future ones.

Equations (12) and (13) 基本是 equations (4) and (5).



#### Backward Fokker-Planck Solution

$$
\begin{align}
-\frac{\partial p\left(x_t, t\right)}{\partial t}= & A \sum_i \frac{\partial}{\partial x_t^i} p\left(x_t, t\right)   -\frac{1}{2} B B' \sum_{i, j, k} \frac{\partial^2 p\left(x_t, t\right)}{\partial x_t^i \partial x_t^j} .
\end{align}
$$

One has
$$
p\left(x_t\right)=\frac{1}{(2 \pi)^{n / 2}|P|^{1 / 2}} \exp \left\{-\frac{1}{2} x_t^{\prime} P^{-1} x_t\right\}
$$
重點應該是 $P$ 應該隨著時間改變。

Here, $P$ is the solution of $P A^{\prime}+A P=-B B^{\prime}$, and is assumed nonsingular. Then
$$
\begin{aligned}
\frac{1}{p\left(x_t\right)} \sum_j \frac{\partial}{\partial x_t^j}\left[p\left(x_t\right) B^{j k}\right] & =-\sum_j \left(P^{-1}\right)^{j i} x_t^i B^{j k} \\
& =-k \text { th entry of } B^{\prime} P^{-1} x .
\end{aligned}
$$
Thus, following (3.10),
$$
\mathrm{d} \bar{w}_{\mathrm{t}}=\mathrm{d} w-B^{\prime} P^{-1} x \mathrm{~d} t
$$









## Appendix

### Appendix A

Because
$$
p\left(x_t, t, x_s, s\right)=p\left(x_s, s \mid x_t, t\right) p\left(x_t, t\right)
$$
we can attempt to obtain a partial differential equation for $p\left(x_t, t, x_s, s\right)$, regarding $x_t, t$ as the independent variables and $x_s, s$ as parameters. We obtain, combining (5.2) through $(5.4)$
$$
\begin{aligned}
\frac{\partial p\left(x_t, t, x_s, s\right)}{\partial t}= & \text { terms involving } f, g, p\left(x_t, t\right) \text { and } p\left(x_s, s \mid x_t, t\right) \\
& \text { and their } x_t \text {-derivatives. }
\end{aligned}
$$
tions is, for $s \geqslant t$
$$
\begin{aligned}
-\frac{\partial}{\partial t} p\left(x_t, t, x_s, s\right)= & \sum_i \frac{\partial}{\partial x_t^i}\left[\bar{f}^i\left(x_t, t\right) p\left(x_t, t, x_s, s\right)\right] \\
& +\frac{1}{2} \sum_{i, i, k} \frac{\partial^2\left[p\left(x_t t, x_s, s\right) g^{i k}\left(x_t, t\right) g^{j k}\left(x_t, t\right)\right]}{\partial x_t^i \partial x_t^i}
\end{aligned}
$$
where $\bar{f}^i$ is as before, viz,
$$
\bar{f}^i\left(x_t, t\right)=f^i\left(x_t, t\right)-\frac{1}{p\left(x_t, t\right)} \sum_{i, k} \frac{\partial}{\partial x_t^i}\left[p\left(x_t, t\right) g^{i k}\left(x_t, t\right) g^{i k}\left(x_t, t\right)\right]
$$
The same partial differential equation (but with different boundary conditions of course) is satisfied by $p\left(x_t, t \mid x_s, s\right)$ [and in fact $p\left(x_t, t\right)$ - this is trivial to see. Just as (5.3) corresponds to the forward model $(5.1)$, so then (5.5) has to correspond to the reverse model

$$
\mathrm{d} \bar{x}_t=\bar{f}\left(\bar{x}_t, t\right) \mathrm{d} t+g\left(\bar{x}_t, t\right) \mathrm{d} \bar{w}_t
$$