---
title: From Quantum Mechanics to Classical Mechanics
date: 2025-11-17 09:28:08
description: revision control
typora-root-url: ../../allenlu2009.github.io
categories:
  - Science
tags:
  - Geometry
  - Math
  - Physics
  - Quantum
---



量子到古典: https://www.youtube.com/watch?v=lJorwy0BQGU
Path Integral: https://www.youtube.com/watch?v=Sp5SvdDh2u8
Free particle path integral 的例子 https://www.youtube.com/watch?v=UWuhMJZaiZM

因爲物理歷史發展的緣故，我們都是先學 classical mechanics, 特別是牛頓力學。再從其中類比出量子力學。但是牛頓力學沒有對應的量子版本。因此大多是從 Lagrangian mechanics 或是 Hamiltonian mechanics 推測對應的量子版本。或是根本就不提量子力學和古典力學的關係。

這個 YouTube 反其道而行。先假設量子力學是完整和 compact 的。從而類別得到古典的 Lagrangian (classical continuous state, L, quantum path integral),  S(L) (對應 path integral) 和 x(t) (最有可能的 path integral),  即是最小作用原理。   一旦得到 x(t) 就可以得到所有其他的古典物理量。

![[Pasted image 20251118001904.png]]


## Quantum Path Integral vs. Classical Path Integral

傳統上，我們從古典的最小作用原理出發 $S = \int \mathcal{L} dt$，然後說同樣原理可以“類比”在量子物理。這是無法推導的，因為不可能從古典力學“推導”出量子力學。

但是用同樣的方法，我們可以從量子物理 (包含量子場論) 的 path integral 出發，推導在古典物理的最小作用原理。注意量子物理的 Feynman Path Integral 等價 Schrödinger Wave Equation 等價 Heisenberg Matrix Mechanics.  所以這裡我們純粹從 "粒子" 角度的 path integral 出發，不涉及 wave equation, 但是巧妙的融入 wave 的概念。

### 什麼是量子物理的 Path Integral?

如下圖從 $t_i\to t_f$ 包含無窮多的 paths.  
* 古典物理只會有一條 path 存在，就是最小作用原理的 path.  
* 但是在量子物理萬物皆有可能，只是機率大小的問題。每一個 path 都是一在無窮多維空間 (Hilbert space) 的向量。可以想像這個向量是把時間切除無窮多段，每一段對應的 $x$ 值就是這個向量的分量。


![[Pasted image 20251119222011.png]]

量子物理 path integral 下一個問題是**如何指定每一個 path 的機率！**
**這不是一個簡單的 linear combination!**  Feynman 天才之處在於提出這個 path integral，融合粒子物理的 wave 和古典物理的作用 S 的關係。
* 每一條 path 對應的 S (古典物理) 都可以視為一個點光源存在 $e^{iS/\bar{h}}$.  **每一個 path 的機率都相等因為 magnitude = 1！**. 
* 這不是非常矛盾？但是把每一個點光源看成波，會和其他的“點光源”干涉產生更有可能或更不可能 path.  一般和附近 path 的干涉才會重要，因為太遠的 path 基本都是不相干。最後 normalize 就可以忽略不計。
* 在古典物理極限 $S >> \bar{h}$,  所以只有在 $\delta S=0$ 的一條 path 會建設相干變成古典的最小作用原理！
* 但是在微觀粒子 $S \approx \bar{h}$, ($10^{-34}$ Joule.second) 原則上所有的 path 都可能，經過 normalization 不同 path 和附近 path 干涉後有不同的機率分佈。

![[Pasted image 20251119223512.png]]![[Pasted image 20251119223922.png]]

## Unit Check

![[Pasted image 20251119230212.png]]
![[Pasted image 20251119230150.png]]

## 一個栗子

但是在微觀粒子 $S \approx \bar{h}$, ($10^{-34}$ Joule.second) 原則上所有的 path 都可能，經過 normalization 不同 path 和附近 path 干涉後有不同的機率分佈。我們要解釋這句話的意思！

Free particle 古典物理是走直線。量子物理是直線為平均值的高斯分佈。

![[Pasted image 20251119231743.png]]
![[Pasted image 20251119231755.png]]

以上是 complex amplitude,  需要取 L2 norm 變成機率。

![[Pasted image 20251119232104.png]]

## Feynman Path Integral 等價 Schrödinger Wave Equation

![[Pasted image 20251119232832.png]]

![[Pasted image 20251119232904.png]]


由上面的例子，從 path integral 推出古典物理的最小作用原理。和量子物理的 wave equation.
Path integral 是所有 path 皆有可能，只是干涉後會有一個分佈。和 wave equation 等價。
## 最小作用原理似乎並非基本原理，而是 wave 干涉的結果
這似乎 make sense, 因爲 least action principle 是 global property 需要上帝視角, wave interference 則是 local property.

### Euler method, 不 robust

![[Pasted image 20251212194755.png]]


### 會 preserve phase space volume!  Hamiltonian mechanics using symplctic geometry.
![[Pasted image 20251212195356.png]]

## 對比一下 machine learning 的 momentum method.  基本是一樣的操作！

$\nabla{f}(x)$ 就是 force 或是加速度。

![[Pasted image 20251212195446.png]]
### 再看一下 DeltaNet,  有點樣子，但不像。
![[Pasted image 20251212201624.png]]


## 量子角動量和 action 的關係？

 $e^{iS/\bar{h}}$. 
 ![[Pasted image 20251227135723.png]]