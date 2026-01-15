---
title: Discrete Diffusion
date: 2025-03-23 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: 2023-02-08-GenAI_Diffusion
categories:
  - GenAI
tags:
  - Diffusion
  - GenAI
  - Graph
  - Language
  - Laplacian
  - Math-AI
---



**Excellent lectures!!!!:**   https://www.youtube.com/watch?v=8mxCNMJ7dHM&list=PL0H3pMD88m8XPBlWoWGyal45MtnwKLSkQ


## Main Reference

https://arxiv.org/pdf/1503.03585.pdf : original Stanford Diffusion paper: very good!

https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ : good blog article including conditional diffusion

https://jalammar.github.io/illustrated-stable-diffusion/  by Jay Alammar, excellent and no math!

[@alammarIllustratedStable2022a] by Jay Alammar, excellent and no math!

[@alammarIllustratedStable2022] by Jay Alammar, excellent and no math!




## Takeaways

Score matching is the key!  等價於 denoise, why?
看 Tweedie's formula!

x~ = x + sigma^2 + blurred score!

![[Pasted image 20250318112923.png]]

![[Pasted image 20250318112839.png]]
![[Pasted image 20250323001649.png]]
Tweedie’s Formula [8]. InEnglish,Tweedie’s Formulastates that the true mean of an exponential family distribution, given samples drawn from it, can be estimated by the maximum likelihood estimate of the samples (aka empirical mean) plus some correction term involving the score of the estimate. In the case of  just one observed sample,the empirical mean is just the sample itself. It is commonly used to mitigate sample bias; if observed samples all lie on one end of the underlying distribution, then the negative score becomes large and corrects  the naive maximum likelihood estimate of the samples towards the true mean.



DDPM vs. DDIM


類似 discrete case. 

DDPM: predict noise using score matching!  如上式
DDIM: predict $x_0$ directly $E[x_0 \vert x_t]$j.  也就是 flow model
CM: consistency mode:  利用 NN 直接 predict ODE output

![[Pasted image 20250322172344.png]]



