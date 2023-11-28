---
title: Connection and Covariant Derivative？
date: 2023-07-09 23:10:08
categories:
- Math
tags: [Manifold, Covariant, Contravariant]
typora-root-url: ../../allenlu2009.github.io


---

https://www.youtube.com/watch?v=cEEahoUUGyc





## Vector Space Abstraction

Key message: covariant derivative (or connection) 是一種 vector space.



<img src="/media/image-20230709092121385.png" alt="image-20230709092121385" style="zoom: 33%;" />



## Two Views

Covariant derivative (extrinsic view with normal component subtracted): Bird-Eye-View (BEV)

Intrinsic view (Bug-Eye-View) BEV ??!



<img src="/media/image-20230709185529006.png" alt="image-20230709185529006" style="zoom: 50%;" />



<img src="/media/image-20230709185010579.png" alt="image-20230709185010579" style="zoom: 33%;" />



<img src="/media/image-20230709223358116.png" alt="image-20230709223358116" style="zoom:33%;" />

Geodesic 就是第一項為 0.



<img src="/media/image-20230709185610745.png" alt="image-20230709185610745" style="zoom:33%;" />





### Intrinsic 用球面為例

External View

<img src="/media/image-20230709185807221.png" alt="image-20230709185807221" style="zoom:33%;" />

For Intrinsic View :  **No more global x,y,z (use u, v),  no global origin, no position vectors, R**

<img src="/media/image-20230709190016054.png" alt="image-20230709190016054" style="zoom:33%;" />

而是：

<img src="/media/image-20230709190129557.png" alt="image-20230709190129557" style="zoom:33%;" />

<img src="/media/image-20230709190251662.png" alt="image-20230709190251662" style="zoom:33%;" />

<img src="/media/image-20230709190327944.png" alt="image-20230709190327944" style="zoom:33%;" />

Covariant Derivative

<img src="/media/image-20230709190418846.png" alt="image-20230709190418846" style="zoom:33%;" />

沒有 normal vector, just ignore!

<img src="/media/image-20230709223136125.png" alt="image-20230709223136125" style="zoom:33%;" />



Need a new strategy to compute Christoffel symbol!!



In intrsinc geometry,  metric tensor needs to be given!!

如果是已知的 manifold, 可以直接計算 metric tensor (from external view).

如果是廣義相對論或其他物理，metric tensor 可以猜出來，or from God.



只要給定 metric tensor, 可以計算 (Intrinsic) Christoffel symbol, 以及 (intrinsic) curvature.

<img src="/media/image-20230709191009204.png" alt="image-20230709191009204" style="zoom:33%;" />

<img src="/media/image-20230709205304274.png" alt="image-20230709205304274" style="zoom:33%;" />







## Connection = Covariant Derivative











<img src="/media/image-20230709092249384.png" alt="image-20230709092249384" style="zoom:33%;" />





<img src="/media/image-20230709092410665.png" alt="image-20230709092410665" style="zoom:33%;" />



<img src="/media/image-20230709205520745.png" alt="image-20230709205520745" style="zoom:33%;" />



<img src="/media/image-20230709205946301.png" alt="image-20230709205946301" style="zoom:33%;" />

<img src="/media/image-20230709210122697.png" alt="image-20230709210122697" style="zoom:33%;" />

* Directional derivative 一般只是 on scalar function/field (i.e. gradient).  可以擴展到 vector function/field 就是 covariant derivative!
* Directional derivative on scalar field 就是梯度，是 covariant!  所以擴展到 vector field 的 covariant derivative 也是 covariant!
* 下面 1, 2 式 a, b是常數。3, 4  f 是 scalar field. 滿足 vector space 的定義。
* 
* $$
  \begin{aligned}
  & \nabla_{a \vec{w}+b \vec{t}} \vec{v}=a \nabla_{\vec{w}} \vec{v}+b \nabla_{\vec{t}} \vec{v} \\
  & \nabla_{\vec{w}}(\vec{v}+\vec{u})=\nabla_{\vec{w}} \vec{v}+\nabla_{\vec{w}} \vec{u} \\
  & \nabla_{\vec{w}}(f \vec{v})=\left(\nabla_{\vec{w}} f\right) \vec{v}+f\left(\nabla_{\vec{w}} \vec{v}\right) \\
  & \nabla_{\partial_i}(f)=\frac{\partial f}{\partial u^i}
  \end{aligned}
  $$

<img src="/media/image-20230709092820300.png" alt="image-20230709092820300" style="zoom:30%;" />





## 



<img src="/media/image-20230709094449895.png" alt="image-20230709094449895" style="zoom: 50%;" />

<img src="/media/image-20230709161855358.png" alt="image-20230709161855358" style="zoom: 67%;" />