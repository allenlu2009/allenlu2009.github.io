---
title: Geometric Algebra (GA) Introduction and Application
date: 2022-12-03 09:28:08
categories: 
- Math
tags: [Geometric Algebra, Vector Calculus]
description: 浮點運算
typora-root-url: ../../allenlu2009.github.io
---



## Introduction

有兩個角度看 Geometric Algebra (GA):  (1) 從歷史角度；(2) 從幾何角度

### 歷史角度： 

* 笛卡爾開創結合幾何和代數的先河，引入笛卡爾坐標系。
* 高斯引入 (2D) 複數平面，成功的賦予複數 $i$ 幾何意義，就是逆時針旋轉 90 度。一個複數平面的 vector $(x,y) = x + iy$， 可以加、減、乘、除一個複數得到新的 vector.  加和減對應 vector 的加法和減法。更驚奇的是乘法和除法。乘或除一個實數對應 **vector 的放大或縮小**。乘或除虛數 $i$ 卻是對應 **vector 逆時針或順時針旋轉 90 度**。如果是乘或除複數就是對應 **vector 的縮放和旋轉 (scale and rotation).**   黎曼進一步解析延拓，開創了複變分析的數學分支。
* 很自然的下一步就是拓展到 (3D) 空間幾何。最初 Hamilton (1843) 的 quaternion 似乎是衆望所歸。不過有幾個問題：
  * The basis vectors square to -1 and not +1, i.e. $i^2 = j^2 = k^2 = ijk = -1$ and $i j = k$    因此很多物理量變得非常奇怪，例如 kinetic energy:  $\frac{1}{2} m \boldsymbol{v}^2 = - \frac{1}{2} m |\boldsymbol{v}|^2$
    * Why 2D 複數沒有這個問題?  因爲 2D 用 complex conjugate $\frac{1}{2} m \boldsymbol{v} \boldsymbol{v}^* = \frac{1}{2} m |\boldsymbol{v}|^2 = \frac{1}{2} m (x^2 + y^2)$
    * 3D quaternion 沒有 complex conjugate?
  * All operations are non-commutative (rotation is non-commutative?)
  * 需要額外加上第四個 scalar component, $a$, 多餘且無意義。
  * **最主要的問題是 quaternion 的 basis vectors, $i, j, k$, 是對應 $x, y, z$ 軸的 90 度旋轉，少了 vector 的 scaling (or projection, or similarity) basis!?**   從 group theory 角度，quaternion 只有 address su(2) symmetry.

<img src="/media/image-20221203214012485.png" alt="image-20221203214012485" style="zoom: 50%;" />



* **因此 Gibbs/Heaviside 的 vector $[x, y, z] = x e_i + y e_j + z e_k$, where $e_i^2 = e_j^2 = e_k^2 = 1$ 成爲主流迄今。**
  * 此時相反變成沒有 rotation 的 basis!  例如 $[x, y, 0]$ 的 vector 無法乘、除一個 vector 產生 $e_k$ 方向的 component!
  
  * Tensor 的 covariant and contravariant coordinate 無法解決這個問題！**Really? TBC.**
  
  * 解決這個問題需要定義新的 vector multiplication, 就是外積或是 curl,  $A \times B$ or $\nabla \times B$ operations.
  
  * Vector 定義：
    * Inner product 内積是一個 scalar: 大小  $\vec{u} \cdot \vec{v} = \|\vec{u}\| \| \vec{v}\| \cos(\theta)$ , 物理意義就是相似度。這是和坐標系無關！
    
      * $\vec{u} \cdot \vec{v} = (a_1 e_1 + b_1 e_2 + c_1 e_3)\cdot(a_2 e_1 + b_2 e_2 + c_3 e_3) = a_1 a_2 +b_1 b_2 + c_1 c_2$ 
    
    * Cross product 是一個 vector:  $\vec{u} \times \vec{v} = (a_1 e_1 + b_1 e_2 + c_1 e_3)\times (a_2 e_1 + b_2 e_2 + c_2 e_3) = (b_1 c_2 - c_1 b_2) e_1 + (c_1 a_2 - a_1 c_2) e_2 + (a_1 b_2 - b_1 a_2) e_3 $  where $e_i \times e_i = 0 $ and $e_i \times e_j = e_k$.      $B \times A =  -A \times B$ 
    
    * Outer product 外積是一個 bi-vector: 方向是第一 vector 到第二 vector 的方向 , 大小是圍的面積： $\vec{u} \wedge \vec{v} = \|\vec{u}\| \| \vec{v}\| \sin(\theta)$， 這是和坐標系無關！ 
    
      * $\vec{u} \times \vec{v} = (a_1 e_1 + b_1 e_2 + c_1 e_3)\times (a_2 e_1 + b_2 e_2 + c_2 e_3) = (b_1 c_2 - c_1 b_2) e_1 + (c_1 a_2 - a_1 c_2) e_2 + (a_1 b_2 - b_1 a_2) e_3 $  where $e_i \times e_i = 0 $ and $e_i \times e_j = e_k$.      $B \times A =  -A \times B$ 
    
    * 在 3D space, 可以把 cross product 和 outer product 用 $\vec{u} \wedge \vec{v} = i \vec{u} \times \vec{v} $ link 起來！
    
    * Caveat: 不像 Gauss 定義的 2D complex plane, 複數可以加、減、乘、除。**以上的 vector space inner product 或 cross-product 沒有逆運算！ 後面會提到 GA product 有逆運算** Tensor 有除法嗎? TBC
    
      
  
* **Clifford 定義  geometric product (幾何積, 乘法): $\vec{u} \vec{v} = \vec{u} \cdot \vec{v} + \vec{u} \wedge \vec{v}$ .  就是内積加上“外積”，這是和坐標系無關！ 可以推導一些特性：**
  
  * $\vec{u} \cdot \vec{v}$  是標準内積。 $\vec{u} \wedge \vec{v}$ 并不是 vector cross product，而是類似 cross product 的 wedge operation.  這是 GA 與衆不同的地方！
    * 外積得到 vector, 類外積得到 bi-vector.
    * 外積在 2D 不存在，類外積等價與複數運算。
  * $\vec{u} \vec{u} = \vec{u} \cdot \vec{u} = \|\vec{u}\|^2 \to \vec{u}^{-1} = \frac{\vec{u}}{\|\vec{u}\|^2}$
  * $\vec{u} \vec{v} = \vec{u} \cdot \vec{v} + \vec{u} \wedge \vec{v} \to \vec{v} \vec{u} = \vec{u} \cdot \vec{v} - \vec{u} \wedge \vec{v}$  這其實很像 conjugate operation.
  * $\vec{u} \cdot \vec{v} = \frac{1}{2}(\vec{u} \vec{v} + \vec{v} \vec{u}) \text{ and } \vec{u} \wedge \vec{v}= \frac{1}{2}(\vec{u} \vec{v} - \vec{v} \vec{u})$  





### GA Math Basis 

我們落實到直角坐標系。Q: 需要限制在直角坐標系嗎? 

* $e_i e_i = 1$ (scalar)  and  定義 bi-vector: $e_i e_j = - e_j e_i$ when $i \ne j$  

* 我們可以檢查 GA 的 vector 乘 vector, 先看 2D.    

* $\hat{x}\hat{y} = i$

$$
\begin{aligned}
\vec{u} =& a_1 \hat{x}+b_1 \hat{y} \\
\vec{v} =& a_2 \hat{x}+b_2 \hat{y} \\
\vec{u} \vec{v} =& \left(a_1 \hat{x}+b_1 \hat{y}\right)\left(a_2 \hat{x}+b_2 \hat{y}\right) \\
=& a_1 a_2 \hat{x} \hat{x}+a_1 b_2 \hat{x} \hat{y} \\
& b_1 a_2 \hat{y} \hat{x}+b_1 b_2 \hat{y} \hat{y} \\
\vec{u} \vec{v} =& a_1 a_2+b_1 b_2+\left(a_1 b_2-b_1 a_2\right) \hat{x} \hat{y} 
\end{aligned}
$$

* 我們可以檢查 GA 的 vector 乘 vector, in 3D

$$
\begin{aligned}
\vec{u} =& a_1 \hat{x}+b_1 \hat{y}+c_1 \hat{z} \\
\vec{v} =& a_2 \hat{x}+b_2 \hat{y}+c_2 \hat{z} \\
\vec{u} \vec{v} =& \left(a_1 \hat{x}+b_1 \hat{y}+c_1 \hat{z}\right)\left(a_2 \hat{x}+b_2 \hat{y}+c_2 \hat{z}\right) \\
=& a_1 a_2 \hat{x} \hat{x}+a_1 b_2 \hat{x} \hat{y}+a_1 c_2 \hat{x} \hat{z} \\
+& b_1 a_2 \hat{y} \hat{x}+b_1 b_2 \hat{y} \hat{y}+b_1 c_2 \hat{y} \hat{z} \\
+& c_1 a_2 \hat{z} \hat{x}+c_1 b_2 \hat{z} \hat{y}+c_1 c_2 \hat{z} \hat{z} \\
\vec{u} \vec{v} =& a_1 a_2+b_1 b_2+c_1 c_2+\left(a_1 b_2-b_1 a_2\right) \hat{x} \hat{y}+ \left(b_1 c_2-c_1 b_2\right) \hat{y} \hat{z}+\left(a_1 c_2-c_1 a_2\right) \hat{x} \hat{z} 
\end{aligned}
$$

* **注意我們沒有讓 bi-vector 變成 vector, i.e.**  $\hat{x} \hat{y} \ne \hat{z}$.  **但是 coefficient 卻是一樣。這是爲什麽 wedge operation 是 "類外積"，但不是外積！** 
* vector 乘 vector 得到 scalar + bivector.  我們可以定義任何 bivector 乘 bivector 或是 vector 得到更多的變化。
* In general
  * 2D multivector:  $a + b \hat{x} + c \hat{y} + d \hat{x}\hat{y}$
  * 3D multivector:  $a + b \hat{x} + c \hat{y} + d \hat{z} + e \hat{x}\hat{y} + f \hat{y}\hat{z} + g \hat{x}\hat{z} + h \hat{x}\hat{y}\hat{z}$

幾個問題?

* 幾何和物理意義



### 幾何角度

另一個是幾何角度，下圖提供簡單的圖示。

* Scalar 基本代表 scale or magnitude；vector 則有 vector scale 和 vector 方向；Bi-vector 的大小是面積也有方向 (是面積的方向，不是 normal 的方向)；Tri-vector 的大小是體積，沒有方向? 
* GA 的加法就是同一類的 component 可以相加，不同類 component 互不干涉。
* GA 的乘法前兩類已經討論過。比較特別的是 trivector 乘 vector 得到 bivector; bivector 乘 bivector 得到 bivector 加上 scalar? Yes.

<img src="/media/image-20221204191713985.png" alt="image-20221204191713985" style="zoom:40%;" />

Spacetime Physics with Geometric Algebra1

Ray tracing: Geometric Algebra for Computer Science

SLAM: Geometric Algebra Applications Vol. I Computer Vision, Graphics and Neurocomputing

NeRF?/NAF

Medical image

## Applications

Spacetime Physics with Geometric Algebra1

Ray tracing: Geometric Algebra for Computer Science

SLAM: Geometric Algebra Applications Vol. I Computer Vision, Graphics and Neurocomputing

NeRF?/NAF

Medical image



## Application in Physics

### Maxwell Equation

首先定義:
$$
\begin{aligned}
\nabla &=\frac{1}{c} \frac{\partial}{\partial t}+\vec{\nabla} \\
\vec{\nabla}&= \frac{\partial}{\partial x} \hat{x}+\frac{\partial}{\partial y} \hat{y}+\frac{\partial}{\partial z} \hat{z} \\
J &=c \rho-\vec{J} \\
F &=\vec{E}+i c \vec{B} \\
\end{aligned}
$$

* $\vec{B}$ is bivector, $i$ 是 trivector, 所以 $i \vec{B}$ 變成 vector 可以和 $\vec{E}$ 一致
* $\rho$ 是 scalar, $\vec{J}$ 是 vector.
* Caveat: $\nabla F = \nabla \cdot F + \nabla \wedge F$

Maxwell Equation 可以簡化成:
$$
\begin{aligned}
&\text{Maxwell Equation: }  \nabla F = \frac{J}{c \varepsilon_0} \\
\Rightarrow  &\left(\frac{1}{c} \frac{\partial}{\partial t}+\vec{\nabla}\right)(\vec{E}+i c \vec{B})=\frac{c \rho-\vec{J}}{c \varepsilon_0}\\
& \vec{\nabla} \cdot \vec{E}+\frac{1}{c} \frac{\partial \vec{E}}{\partial t}+i c \vec{\nabla} \wedge \vec{B}+\vec{\nabla} \wedge \vec{E}+i \frac{\partial \vec{B}}{\partial t}+i c \vec{\nabla} \cdot \vec{B}=\frac{\rho}{\varepsilon_0}-\frac{\vec{J}}{c \varepsilon_0} \\
\Rightarrow & \text{ (scalar, Gauss Law) }\vec{\nabla} \cdot \vec{E}=\frac{\rho}{\varepsilon_0} \\
& \text{ (vector, Ampere Law) } \frac{1}{c} \frac{\partial \vec{E}}{\partial t}+i c \vec{\nabla} \wedge \vec{B}=-\frac{\vec{J}}{c \varepsilon_0} \longrightarrow \vec{\nabla} \times \vec{B} - \mu_0 \varepsilon_0  \frac{\partial \vec{E}}{\partial t} =\mu_0 \vec{J} \\
& \text{ (bivector, Faraday Law) } \vec{\nabla} \wedge \vec{E}+i \frac{\partial \vec{B}}{\partial t}=0  \longrightarrow  \vec{\nabla} \times \vec{E}+ \frac{\partial \vec{B}}{\partial t}=0\\
& \text{ (trivector) } i c \vec{\nabla} \cdot \vec{B}=0  \longrightarrow \vec{\nabla} \cdot \vec{B}=0 \\
\end{aligned}
$$



### Rotor and Spinor



## GA 的缺點

似乎很明顯，

* 就是 dimension 太多： 2D 還好，只有 4 dimension: 1 (scalar) + 2 (vectors) + 1 (bivector, scalar)

3D 需要 8 dimension:  1 (scalar) + 3 (vectors) + 3 (bivector) + 1 (trivector, scalar)

 = 8.  如果 4D 需要 16 dimension.

* How to reconcile the $i$ difference between 2D and 3D?  No problem at all,  2D 的 $i$  就是 $\hat{e}_1 \hat{e}_2 = - \hat{e}_2 \hat{e}_1 \to (\hat{e}_1 \hat{e}_2)^2 = \hat{e}_1 \hat{e}_2 \hat{e}_1 \hat{e}_2 = - \hat{e}_1 \hat{e}_2 \hat{e}_2 \hat{e}_1  = -1$

*  3D 的 $i = e_1 e_2 e_3$  = $(\hat{e}_1 \hat{e}_2 \hat{e}_3)^2 = \hat{e}_1 \hat{e}_2 \hat{e}_3 \hat{e}_1 \hat{e}_2 \hat{e}_3 = - \hat{e}_1 \hat{e}_2 \hat{e}_3 \hat{e}_1 \hat{e}_3 \hat{e}_2   = \hat{e}_1 \hat{e}_2 \hat{e}_3 \hat{e}_3 \hat{e}_1 \hat{e}_2 = (\hat{e}_1 \hat{e}_2)^2 = -1$

  

## Reference

Geometric Algebra: very good!!!

https://www.youtube.com/watch?v=60z_hpEAtD8&ab_channel=sudgylacmoe

Tensor Calculus: MathTheBeautiful

3D case of Geometric algebra?

https://www.youtube.com/watch?v=e0eJXttPRZI&ab_channel=MathTheBeautiful



Covariance and Contra-variance of vector/tensor?



Unify covariance and contra-vairance of vector/tensor??

in Allen Lu MWEB:  相對論和張量分析 - Coordinate Covariant, Contravariant, Invariant (座標系協變，逆變，不變)



1-form and ? -form tensor calculus?

In Allen Poincaré conjecture/theorem and Ricci flow

Tensor calculus:  Allen Lu MWEB: and Youtube maththebeautifu!!

https://www.youtube.com/watch?v=e0eJXttPRZI&ab_channel=MathTheBeautiful

# Poincare Conjecture/Theorem and Ricci Flow





## Differential geometry vs. Tensor Analysis vs. Geometric Algebra



Differential geomery base: 1. 座標無關，2. 曲面，3. Einstein symbol (covariant and contra variant == vector and 1-form???)

Tensor analysis: 1. 座標無關，2. 平面(?)，3. Einstein symbol (vector and 1-form)

Geometric Algebra:  1. tightly coupled with 座標?  2. 平面 (no)?  



我們從問題出發

1. Tensor analysis/calculus 和 geometric algebra, 到底誰 cover 誰?
2. Tensor 的基礎是座標不變性。在 geometric algebra 對應是什麼?
3. Covariant and Contravariant 對應 geometric algebra 什麼?
4. vector vs. 1-form, vector 對應 geometric algebra 的 2D, 1-form 對應什麼?
5. Einstein symbol 在 geometric algebra 有表現嗎?
6. 目前都是平面，在曲面或是 manifold, 那一個更 general?
7. Manifold 也有 geometric algebra 嗎?  exp map and tagent plane 有對應的表現嗎?
8. Maxwell queation in different forms.  General relativity in different forms!  



[(4) David Hestenes - Tutorial on Geometric Calculus - YouTube](https://www.youtube.com/watch?v=ItGlUbFBFfc&ab_channel=NomenNominandum)

![image-20221202225211143](C:\Users\allen\OneDrive\allenlu2009.github.io\media\image-20221202225211143.png)