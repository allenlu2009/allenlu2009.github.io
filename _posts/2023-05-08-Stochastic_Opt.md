---
title: Math AI - Optimization II 
date: 2023-05-01 23:10:08
categories:
- Math_AI
tags: [SDE, Stochastic equations, diffusion equations, reverse-time equations, Fokker-Planck equations]
typora-root-url: ../../allenlu2009.github.io
---



## Source

YouTube



## Background

Stochastic gradient descent



* Noisy Gradient
* Stochastic Optimization
* Random coordinate descent!





## Noisy gradients



Instead of $\nabla f $,  我們得到的是 $E[g] = \nabla f$





## Stochastic Optimization

問題描述

$\min: f(x)$,  st:  $x \in C$ 改成

$\min_x: E_\xi [f(x); \xi]$,  st:  $x \in C$ 

什麽是 $\xi$, 就是 machine learning 的 training parameter. 

例如在 linear regression  $\min_x E[(y - \xi^T x)^2]$

$(y_1, \xi_1), (y_2, \xi_2), ...$ 就是 training data.

Empirical Risk Minimization Z(ERM) = $\min f(x) = \frac{1}{n} \sum f_i(x)$

在 classification 問題：

<img src="/media/image-20230508212823246.png" alt="image-20230508212823246" style="zoom: 67%;" />



**NO,  應該反過來。$x$ 最後是 training parameter 可以使用 gradient descent 得到。 $\xi$ 則是 data with distribution.** 





Full GD is very expensive!  going over all data points.



## SGD

GD 有 self-tuning 特性

sub-GD 沒有 self-tuning 特性



在 SGD 也是一樣？yes



Sub-gradient proportional to 1/sqrt(T)

Gradient descent proportional to 1/T 



SGD   1/sqrt(T)

<img src="/media/image-20230509205428327.png" alt="image-20230509205428327" style="zoom:50%;" />



SGD is similar to sub-differential!!

No self-tuning for SGD

<img src="/media/image-20230509210623909.png" alt="image-20230509210623909" style="zoom:67%;" />





