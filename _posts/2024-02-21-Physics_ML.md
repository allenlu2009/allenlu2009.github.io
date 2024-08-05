---
title: Physics Informed ML/AI
date: 2024-03-03 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
description: LLM Tokenizer
typora-root-url: ../../allenlu2009.github.io



---





## Source

* [Physics Informed Machine Learning: High Level Overview of AI and ML in Science and Engineering (youtube.com)](https://www.youtube.com/watch?v=JoFW2uSd3Uo&ab_channel=SteveBrunton)

* from U. of Washington good video!



# Takeaway

* Physics 最重要的是 symmetry, conservation law, invariance 如何體現在 AI/ML!!

  * Astrology vs. Astronomy
  * Alchemy vs. Chemistry 

* Parsimony!

* 最重要是學習如何 embed prior knowledge to ML/AI

* 如何滿足 constraint?  (例如 token translation 問題?)

  1. 使用 loss function, NOT a good way 因爲不會是 0

  2.  explicitly 使用 。。。。 TBD

     <img src="/media/image-20240303204634948.png" alt="image-20240303204634948" style="zoom:33%;" />

* 如何減少 error?

  * 加上 extra physic loss function in training!!   特別是 physic quantity 都是**可微分**的！！！！！
  * <img src="/media/image-20240303204537396.png" alt="image-20240303204537396" style="zoom:50%;" />



## 開場

* Physics 也是一種 (**differentiable) optimization.**
* AI/ML 是 optimization



Physics to enforce ML/AI.

反之 use AI/ML to discover new physics

```python

```



## 煉丹五部曲

<img src="/media/image-20240303202028724.png" alt="image-20240303202028724" style="zoom:50%;" />

加上 Physics!

<img src="/media/image-20240303202208650.png" alt="image-20240303202208650" style="zoom:50%;" />



## 預備知識

<img src="/media/image-20240303202332434.png" alt="image-20240303202332434" style="zoom:67%;" />



<img src="/media/image-20240303202537628.png" alt="image-20240303202537628" style="zoom:80%;" />



### Data Collection (Huge advantage using physical knowledge 可以大幅減少 data!!)

1. use symmetry (圖1)

2. use right coordinate (圖 2)

3. use Fourier transform ... (可以視爲 new coordinate system)

4. Simulation (slow, rich in spatial) or experiment (fast, good for temporal) (圖 3)

5. Data generalization 需要 滿足 physics (symmetry, conservation), and parsimony!

   <img src="/media/image-20240303230047484.png" alt="image-20240303230047484" style="zoom:50%;" />

<img src="/media/image-20240303230114550.png" alt="image-20240303230114550" style="zoom:50%;" />

<img src="/media/image-20240303230338398.png" alt="image-20240303230338398" style="zoom:50%;" />

<img src="/media/image-20240303230942777.png" alt="image-20240303230942777" style="zoom:50%;" />







## Symmetry,  Invariance, Equivariance

<img src="/media/image-20240309074721679.png" alt="image-20240309074721679" style="zoom:67%;" />



常見的對稱性。

<img src="/media/image-20240407133600878.png" alt="image-20240407133600878" style="zoom:80%;" />



#### 三種常見方法使用對稱性

1. Promoting 對稱性：加上 loss function.   M 是 manifold.  第二項是 y_hat 和 manifold 的距離要 minimize.

   <img src="/media/image-20240407140317441.png" alt="image-20240407140317441" style="zoom:50%;" />

2. Enforce 對稱性：直接放在 optimization procedure （類似 chain of thought?）中： KKT condition,  上述矩陣對稱保持。直接限制 y_hat 要在 manifold 上。這是 constrained optimization！

   <img src="/media/image-20240407140420102.png" alt="image-20240407140420102" style="zoom:50%;" />

3. Discovery 對稱性：反過來可以用 ML/AI 發現對稱性。

   







## Appendix
