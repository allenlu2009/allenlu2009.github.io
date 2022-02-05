---
title: Thermal Resistance
date: 2022-01-17 09:28:08
categories: 
- AI
tags: [CV, Optical Flow]
typora-root-url: ../../allenlu2009.github.io
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>


Reference

[803PET21.pdf (unipi.it)](http://www.iet.unipi.it/f.baronti/didattica/CE/Files/803PET21.pdf)



首先，JEDEC (Joint Electron Tube Engineering Council) 的標準，與熱相關的標準主要有兩個

－JESD51系列：包括大多數IC等封裝的“熱”相關規格。
－JESD15系列：對模擬用的熱阻模型進行規格化。



#### IC 散熱以及 Thermal Resistance

上面介紹過熱量，溫度，比熱，thermal conductivity, thermal diffusivity.  此處順便提一下 IC 散熱常用的 thermal resistance [nelsonPackageThermal2018].   另文再詳細介紹。

IC 和散熱分成幾個節點：

* Die 對應的是 $T_j$ , junction temperature: 熱源 (heat generator), 就是下圖淺藍色的 die 和紅色的 $T_j$.
* Package 對應的是 $T_c$ , IC top case temperature:  就是下圖灰色部分。這是散熱 path 1, 從 package 再散熱到 ambient (air) $T_a$.
* PC board 對應的是 $T_b$,  PCB (surface) temperature: 就是下圖深藍色部分。這是散熱 path 2, 從 package (power/ground) balls 經 PCB traces, 最後在散熱到 ambient $T_a$.    

<img src="/media/image-20220122230319659.png" alt="image-20220122230319659" style="zoom: 67%;" />

可以定義如下的 (theta) 熱阻 (thermal resistance).  並利用熱源和熱阻組成的 ”熱路" 表示散熱的路徑如下。就如同電源和電阻組成電路的概念。

$$
\begin{aligned}
&\Theta_{\mathrm{JA}} : \text{Junction-to-Ambient Thermal Resistance}\\
&\Theta_{\mathrm{JB}} : \text{Junction-to-Board Thermal Resistance}\\
&\Theta_{\mathrm{JC}} : \text{Junction-to-Case Thermal Resistance}\\
\end{aligned}
$$

<img src="/media/image-20220123220057041.png" alt="image-20220123220057041" style="zoom:80%;" />

我們看一些 package 的 thermal resistance 的例子。就是 1W power 會造成溫度上升幾度。愈大就代表阻值愈大。

<img src="/media/image-20220122221909927.png" alt="image-20220122221909927" style="zoom: 80%;" />





From [mengThermalModel2013] and  [parkAccuratePrediction2018], 我們可以 model thermal flow as first order RC network. 

這似乎和 "thermal resistance" 的概念相抵觸。因爲 resistor 是 instant response, 沒有 time constant 的概念。

Time constant 是由 RC network 構成。但是我們有 thermal capacitance?

It turn out 在電路理論，我們假設 quasi-static condition, 也就是 circuit dimension << operating wavelength.  所以可以假設 resistance 是 instant.  但是在 thermal network, 因爲 thermal spread speed 遠小於光速。因此會有 time constant 反應時間。而當使用 finite element method (FEM) 解 heat PDE 時，可以用 first order RC network with thermal resistor and thermal capacitor 得到。  

<img src="/media/image-20220126225302477.png" alt="image-20220126225302477" style="zoom:80%;" />



## Reference