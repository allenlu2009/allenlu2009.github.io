---
title: Computer Vision - HDR Network
date: 2021-12-02 11:10:08
categories: 
- AI
tags: [Autoencoder, Unet, HDR, HDRNet, Bilateral filter]
typora-root-url: ../../allenlu2009.github.io
---


<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>



## High Dynamic Range, Why

一般人的聽覺或視覺對於 noise 遠比 dynmaic range 敏感。如果聽到音樂有雜音，或是影像有雜訊，基本無法接受。相反如果音樂或影像 dynamic range 比較差，人耳或是人眼一般是很 tolerable.  In a sense, 這也代表人腦對於對於 low dynamic range 可以經由腦補而得到該有的資訊。但對於雜訊比較無法處理，或是要更努力腦補才能得到資訊。



<img src="/media/image-20211202223319146.png" alt="image-20211202223319146" style="zoom:60%;" />



<img src="/media/image-20211202223416064.png" alt="image-20211202223416064" style="zoom:100%;" />



沒有比較就沒有傷害。如果習慣了看 HDR 的 image or video (下圖左), 再比較 SDR 的 image or video (下圖右), 人眼就可以分辨出差異。而且看久了 HDR 就會覺得回不去 SDR.

<img src="/media/image-20211202223518492.png" alt="image-20211202223518492" style="zoom:80%;" />



HDR vs. SDR (or LDR) 有時不只是美感的問題 (如上圖)，而是 information loss 的問題。此時大腦就需要努力腦補而覺得 SDR/LDR 的影像無法接受 (如下圖)。 



<img src="/media/image-20211202222300301.png" alt="image-20211202222300301" style="zoom:80%;" />



## High Dynamic Range, How

HDR 分為 traditional CV algorithm,  deep learning (DL or AI) algorithm, and hybrid algorithm.

最簡單的做法 (還不到 algorithm) 是 apply 一個 **global tone curve** 把 dynamic range 拉開，如下圖。

這招廣泛用於聲音和影像，例如常聽到的 gamma correction.  這算是小學程度。

<img src="/media/image-20211202225705177.png" alt="image-20211202225705177" style="zoom:100%;" />



這種 global tone curve (or tone map) 不考慮 content 或是 spatial location.  很自然的延伸就是 (1) 根據 content 改變 tone curve, 例如黑夜拉大 dynamic range；大太陽下可能要壓縮 dynamic range;  (2) 根據 local content 分別調整 tone curve.  例如把影像 (or display) 分為 64/128/.../1024 區，每一個小區塊分別有自己的 local tone curve, 根據 local conent 調整。這是目前 TV 或是 mini LED display 的做法。當然每一個區塊不是完全獨立決定，需要考慮上下左右區塊的 tone curve, 以免整體的 tone curve 突兀。





最極緻的 HDR 是每一個 pixel based on local content 都有自己的 tone curve!  





我們從 traditional CV algorithm, based on bilateral filter 看起。