---
title: Generation via Flow Model
date: 2025-03-06 09:10:08
typora-root-url: ../../allenlu2009.github.io
link: 2021-09-19-Deterministic_Probabilistic
categories:
  - GenAI
tags:
  - Fisher
  - Flow-Model
  - GenAI
  - Information
  - Math-AI
  - Math_AI
  - Score
---



[[2021-09-19-Deterministic_Probabilistic]]
## Introduction

Autoregressive 和 variational autoencoder 都各有優缺點。
Flow model 的 key 是 reciprocal!  如此可以容易得到 posterior distribution!


![[Pasted image 20250306152839.png]]


![[Pasted image 20250307145909.png]]

![[Pasted image 20250307153501.png]]
![[Pasted image 20250307145810.png]]
## How to Reciprocal!  Easy, make the neural network deterministic!!!!

為什麼？因為 deterministic function 的反函數只是一個值 (或是幾個值，不過我們會限制 domain 在 1-1 mapping!)  很容易 reciprocal (特別是用 neural network fit)

但是 conditional $p(x\vert z)$  的“反函數" 或是 posterior $p(z \vert x)$  卻是要 evaluate 所有的 $z$ by Bayesian
$p(z\vert x) = p(x \vert z) p(z) p(x)^{-1}$ 因為皆有可能。

缺點是 $z$ 和 $x$ 有相同的 dimensionality!!

![[Pasted image 20250306160402.png]]


## Reference

Ermon youtube cs236: https://www.youtube.com/watch?v=m6dKKRsZwBQ&list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8&index=7&ab_channel=StanfordOnline

