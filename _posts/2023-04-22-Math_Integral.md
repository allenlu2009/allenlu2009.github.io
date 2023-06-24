---
title: Math - 積分 
date: 2023-04-16 23:10:08
categories:
- Math_AI
tags: [SDE, Diffusion]
typora-root-url: ../../allenlu2009.github.io
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>


## Reference

Lebesgue integral

https://en.wikipedia.org/wiki/Lebesgue_integration @wikiLebesgueIntegration2023: Lebesgue 積分英文

https://zh.wikipedia.org/zh-tw/%E5%8B%92%E8%B2%9D%E6%A0%BC%E7%A9%8D%E5%88%86 : 中文

@shionStochasticDynamic2021

https://zhuanlan.zhihu.com/p/343129740

[@ccOneArticle23]. https://zhuanlan.zhihu.com/p/589106222  very good article!!!  SDE for diffusion score





#### 直觀解釋 黎曼 (Riemann) 積分 and 勒貝格 (Lesbegue) 積分

要直觀地解釋兩種積分的原理，可以假設我們要計算一座山在海平面以上的體積。

黎曼積分是相當於把山分為每塊都是一平方米大的方塊，測量每個方塊正中的山的高度。每個方塊的體積約為1x1x高度，因此山的母體積為所有高度的和。

勒貝格積分則是為山畫一張**等高線圖**，每根等高線之間的高度差為一米。每根等高線內含有的岩石土壤的體積約等於該等高線圈起來的面積乘以其厚度。因此母體積等於所有等高線內面積的和。

<img src="/media/image-20230416162150383.png" alt="image-20230416162150383" style="zoom: 67%;" />

佛蘭德（Folland）描述黎曼積分跟勒貝格積分的不同，以非負函數 $ f:[a,b]\mapsto [0,\infty ],\;a.b\in \mathbb {R} $ 這例子來講，黎曼積分是分割 $x$-軸上的定義域區間 為更小的子區間，並計算黎曼和，當子區間越來越小時黎曼和的極限就是黎曼積分；而勒貝格積分則是將 $f$ 在 $y$-軸上的對應域分割成不相交的區間 $\{I_{j}\}_{j=1}^{n}$，並用定義域中的子集合 $\{f^{-1}(I_{j})=E_{j}\}$ 來定義趨近 f 的簡單函數

<img src="/media/image-20230416162253169.png" alt="image-20230416162253169" style="zoom:67%;" />

#### 白話就是：**黎曼積分是分割定義域來計算積分；勒貝格積分則是用分割值域來計算積分**。

* 分割值域有什麽好處?  因爲計算定義域的 “總長度" 或是 ”總面積" 可以用 ”測度理論“。比起分割定義域可能會無法計算個別**無窮小定義域**的 ”長度" 或是 "面積"。 
* 最初測度理論是用來對歐幾里得空間中直線的長度，以及更廣義地，歐幾里得空間的子集的面積和體積進行仔細分析發展出來的。它尤其可以為 $\R$ 的哪些子集擁有長度這個問題提供一個系統性的回答。
* 對於 bouned function $f$ defined on [a,b],  if $f$ is Riemann integrable, then $f$ is Lebesgue integrable.

我們看一個例子: 有理數的指示函數 ![{\displaystyle 1_{\mathbb {Q} }(x)={\begin{cases}1,x\in {\mathbb {Q} }\\0,x\notin {\mathbb {Q} }\end{cases}}\quad }](https://wikimedia.org/api/rest_v1/media/math/render/svg/fae0db3d845f10dec807b429ad7e692b58e85220)是一個無處連續的函數。

- 在區間![[0,1]](https://wikimedia.org/api/rest_v1/media/math/render/svg/738f7d23bb2d9642bab520020873cccbef49768d)之間![1_{\mathbb {Q} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/7ebddb34e2bf4bccf0089c4583a89bd796b05996)**沒有黎曼積分**，因為在實數中有理數和無理數都是稠密的，因此不管怎樣把![[0,1]](https://wikimedia.org/api/rest_v1/media/math/render/svg/738f7d23bb2d9642bab520020873cccbef49768d)分成子區間，每一個子區間裡面總是至少會有一個有理數和一個無理數，因此其[達布積分](https://zh.wikipedia.org/wiki/达布积分)的上限為1，而下限為0。
- 在區間![[0,1]](https://wikimedia.org/api/rest_v1/media/math/render/svg/738f7d23bb2d9642bab520020873cccbef49768d)內![1_{\mathbb {Q} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/7ebddb34e2bf4bccf0089c4583a89bd796b05996)**有勒貝格積分**。事實上它等於有理數的[指示函數]，因為![\mathbb {Q} ](https://wikimedia.org/api/rest_v1/media/math/render/svg/c5909f0b54e4718fa24d5fd34d54189d24a66e9a)是[可數集](https://zh.wikipedia.org/wiki/可數集)，因此 ![\int _{[0,1]}1_{\mathbb {Q} }\,d\mu =\mu (\mathbb {Q} \cap [0,1])=0](https://wikimedia.org/api/rest_v1/media/math/render/svg/9bf52997897b47a982210a92bb1ac615514989bc)



簡單說：**勒貝格積分可以處理很多極限的積分**。




