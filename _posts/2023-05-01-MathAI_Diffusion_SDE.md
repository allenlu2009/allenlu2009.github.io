---
title: Math AI - Diffusion vs. SDE 
date: 2023-05-01 23:10:08
categories:
- Math_AI
tags: [SDE, Stochastic equations, diffusion equations, reverse-time equations, Fokker-Planck equations]
typora-root-url: ../../allenlu2009.github.io
---


## Reference

[2011.13456.pdf (arxiv.org)](https://arxiv.org/pdf/2011.13456.pdf) :  good reference for Diffusion to SDE from Stanford

[PII: 0304-4149(82)90051-5 (core.ac.uk)](https://core.ac.uk/download/pdf/82826666.pdf)  @andersonReversetimeDiffusion1982 : 專門針對 reverse time SDE equations. 

http://pordlabs.ucsd.edu/pcessi/theory2019/gardiner_ito_calculus.pdf

Ito equations



## Introduction



全部的重點就是下圖。用一個神經網路逼近 score function!  因爲我們不知道 $p(x)$

<img src="/media/image-20230501224624557.png" alt="image-20230501224624557" style="zoom:80%;" />

## Background

Method 1: denoising Score Matching with Langevin Dynamics (**SMLD**)

主要是利用 neural network 去近似 p(x) 的 score function,  i.e. gradient of log likelihood.

<img src="/media/image-20230503213753232.png" alt="image-20230503213753232" style="zoom:80%;" />

<img src="/media/image-20230503214401928.png" alt="image-20230503214401928" style="zoom:80%;" />



Method 2: denoising diffusion probabilistic model (**DDPM**)

<img src="/media/image-20230503214423165.png" alt="image-20230503214423165" style="zoom:80%;" />

<img src="/media/image-20230503214453770.png" alt="image-20230503214453770" style="zoom:80%;" />



## SDE and Reverse SDE

Forward SDE

<img src="/media/image-20230503215110115.png" alt="image-20230503215110115" style="zoom:80%;" />



Backward SDE

<img src="/media/image-20230503215131764.png" alt="image-20230503215131764" style="zoom:80%;" />





<img src="/media/image-20230503222636700.png" alt="image-20230503222636700" style="zoom:80%;" />



### DDPM (Variance Preserving SDE)

<img src="/media/image-20230503224837519.png" alt="image-20230503224837519" style="zoom:80%;" />

### SMLD (Variance Exploding SDE)

<img src="/media/image-20230503224822757.png" alt="image-20230503224822757" style="zoom:80%;" />



### Sub-Variance Preserving SDE

<img src="/media/image-20230503225442004.png" alt="image-20230503225442004" style="zoom:80%;" />



