---
title: Deep Learning using Nonequilibrium Thermodynamics
date: 2023-02-07 23:10:08
categories:
- Language
tags: [Graph, Laplacian]
typora-root-url: ../../allenlu2009.github.io

---



## Main Reference

https://arxiv.org/pdf/1503.03585.pdf : original Stanford Diffusion paper:  有點硬核，but very good!

https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ : good blog article including conditional diffusion

https://mbd.baidu.com/newspage/data/landingsuper?rs=2863188546&ruk=xed99He2cfyczAP3Jws7PQ&isBdboxFrom=1&pageType=1&urlext=%7B%22cuid%22%3A%22_i-z80aLH8_cPv8Zla2higiavighaHiUgaSB8gidviKX0qqSB%22%7D&context=%7B%22nid%22%3A%22news_9102962014405338981%22,%22sourceFrom%22%3A%22bjh%22%7D  : excellent article!!

https://jalammar.github.io/illustrated-stable-diffusion/  by Jay Alammar, excellent and no math!



## Introduction



4 種常見的 image generative model.  常見的 trade-off

GAN:  fast inference, but not easy to converge and strange quality

VAE: fast inference and easy train to converge, quality issue

Diffusion:  easy to converge, good quality, but slow for inference

Flow-based models: 

| Generative | Training              | Inference | Quality                     |
| ---------- | --------------------- | --------- | --------------------------- |
| GAN        | Difficult to converge | Fast      | Significant percentage fail |
| VAE        | Easy to converge      | Fast      | Blur quality                |
| Flow       |                       |           |                             |
| Diffusion  | Easy to converge      | Slow      | Good quality                |





<img src="/media/image-20230208221524515.png" alt="image-20230208221524515" style="zoom:80%;" />





## Diffusion Misconception



我對於 diffusion model 的第一個問題是 forward path 加 Gaussian noise 會讓 entropy 增大。這應該是 non-reversible process.

為什麼可以 learning the reverse process?  

learning a noise predictor?  這有點顛覆三觀，特別對於通訊背景的同學！



Ref 提供這樣的 insight:  forward path 要保持在 non-equilibrium process (by time step), 才有機會 learning back.



## Diffusion Path



1. probability (Bayesian)
2. Entropy
3. 

前文討論 diffusion model, 如何從 VAE 演變而來。

Diffusion model 有三條路：(1) denoise; (2) noise prediction; (3) score function reduction?





learning a noise predictor?  這有點顛覆三觀，特別對於通訊背景的同學！



Ref 提供這樣的 insight:  forward path 要保持在 non-equilibrium process (by time step), 才有機會 learning back.