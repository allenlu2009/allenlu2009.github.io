---
title: Neural Network Optical Flow 算法
date: 2022-01-05 09:28:08
categories: 
- Language
tags: [NN, Optical Flow]
typora-root-url: ../../allenlu2009.github.io
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>


## Optical Flow Different Approach

Flow 是**時間和空間連續或整體行爲 (global or continuous behavior over space and time**).   我們不會只看水中**單點瞬間**的運動判斷這是一個 flow or wave, 因爲可能是水流，水波，甚至渦流。因此需要分析時間和空間連續或整體行爲。如何進行? 



有幾種方式：

1. 利用連續性 (bottom-up approach):  用偏微分方程描述每一點附近空間和時間的關係。藉著解偏微分方程 (PDE)，可以得到整體的行爲。**一般的 CV (computer vision) approach, 也就是 analytic formula base approach, 可以直接 leverage PDF 的解法。**相反 AI approach 崇尚 end-to-end learning, i.e. input image and output motion map, 很少利用 PDE 或是 embedded in PDE. 

2. 把 PDE 轉換成積分形式，例如 Maxwell PDE equation 的積分形式。此時已經不是只描述局部行為，而是一個區域或是整體的行為。一般積分形式是為了闡述物理意義，直接求解很困難。**但如果可以把解 PDE 轉換成 (積分) cost function optimization，就可能可以利用 AI approach for end-to-end learning：(1) 利用 deep neural network 近似這個 cost function; (2) 使用 real data training for optimization 收斂到近似解。**當然在適當的 formulation 之下，CV approach 還是有機會利用積分形式。

3. 直接描述整體行爲 (up approach)：例如電路學用 KVL, KCL (Kirchikov Voltage/Current Laws) 簡化代替 Maxwell equations，並且定義電阻、電容、電感等，取代 local transport equation.   或者用熱力學代替統計力學，電子學代替固態物理學。雖然只能得到簡化的 high level solution.   不過在很多實際應用，這樣 high level solution 已經足夠，例如電子電路分析，或是 IC 散熱分析。因為已經是 high level solution, 基本就和 CV or AI approaches 沒有什麼關係。

   

**CV 和 NN optical flow 算法的表面差異，除了上述 CV 比較傾向 PDE base; NN 比較傾向 end-to-end cost function optimization.**  

**真正更本的差異：CV 多半是 fixed formula (e.g. Honk and xx) 和 data 無關；最多有 parameter 可以在 post processing 人爲 tunning base on data;  AI optical flow 的 neural network 是由 data training 得出，所以完全是 data dependent and determined.   Supposely, AI optical flow 的結果更 fit expectation.  特別是在 ill-posed condition.**    



### Optical Flow PDE

本文 focus on PDE 因爲可以看到更基本的物理意義。如上所述，要描述每一點附近空間和時間的關係，需要定義 

* 每一點 (空間和時間)要描述的物理量, e.g. (scalar) density, driving force, (vector) flux/motion; 

* 物理量和時間的關係, continuity equation; 

* 物理量對空間的關係, transport equation;

* combine continuity equation and transport equation 得到完整的 flow or wave PDE.



(I) 空間和時間的物理量是亮度, $I(\mathbf{x}, t)$, where $\mathbf{x}$ 是 2D 的座標，如果 3D 則是 scenic flow. 

(II) **Continuity equation, 基於 brightness/intensity constancy (or conservation?).**

$$
\begin{equation}
\frac{d I}{d t} = \frac{\partial I}{\partial t}+\nabla I \cdot\boldsymbol{v}= \frac{\partial I}{\partial t}+ I_x v_x + I_y v_y \approx 0 \label{OpticFlow}
\end{equation}
$$

**(III) 的問題是 optical flow 似乎沒有 transport equation.**  無法如同 diffusion equation 轉換成一個 scalar 的 Laplacian equation.  可以求解 given the boundary condition.

**$\eqref{OpticFlow}$ 包含 two unknowns, $\boldsymbol{v} = (I_x, I_y)$, 卻只有一個 scalar equation (constraint)。也就是有無窮多 $\boldsymbol{v} = (I_x, I_y)$ 滿足 $\eqref{OpticFlow}$。另外$\eqref{OpticFlow}$ 還 suffer from $I_x=0, I_y=0, \text{ or } \nabla I=0$ (gradient vanishing area) 以及 aperture problem. 這是 ill-posed 問題。**

CV optical flow 算法無法直接解 ill-posed problem, 需要引入額外的 constraint: 例如 minimal energy 或是 smoothness motion.   CV optical flow 分為 coarse and dense optical flow, 此處不討論。其實 AI optical flow 也引入類似的 constraint.   **再次強調，CV optical flow 算法是以 math formula 為主，和 data 無關。  AI optical flow 最重要的差別是用 data training, 所以會更接近 expectation.  本文的重點在 AI or NN optical flow 算法。**


### Optical Flow 積分 Optimization Form

Deep learning 可以用來處理 optical flow 等 ill-posed 問題 by using **the global (observation) cost function** 如下。$f_1$ 是 penalty function, 通常用 $L_2$ Norm.  不過 "softer" penalty function 可能對於 boundary condition 更好。  

$$
\begin{equation}
\mathscr{H}_{\mathrm{obs}}(I, \boldsymbol{v})=\iint_{\Omega} f_{1}\left[\nabla I(\boldsymbol{x}, t) \cdot \boldsymbol{v}(\boldsymbol{x}, t)+\frac{\partial I(\boldsymbol{x}, t)}{\partial t}\right] \mathrm{d} \boldsymbol{x} \label{IntOptic1}
\end{equation}
$$

如前所述，single scalar observation term 無法 estimation 兩個方向的速度 $(v_x, v_y)$。 通常在 cost function 再加上額外的限制 as regularization term.  **額外的限制一般是 enforces a spatial smoothness coherence of the optical flow vector field 如下。這是為了要彌補 optical flow 缺乏 spatial transport equation 的關係？**

$$
\begin{equation}
\mathscr{H}_{\text {reg }}(\boldsymbol{v})=\iint_{\Omega} f_{2}[|\nabla u(\boldsymbol{x}, t)|+|\nabla v(\boldsymbol{x}, t)|] \label{IntOptic2}
\end{equation}
$$

如同 $f_1$ penalty function, the penalty function $f_{2}$ was taken as a quadratic in early studies, but a softer penalty is now preferred in order not to smooth out the natural discontinuities (boundaries,...) of the velocity field.

Based on Eqs. $\eqref{IntOptic1}$ and $\eqref{IntOptic2}$ , the estimation of motion can be done by minimizing:

$$
\begin{equation}
\begin{aligned}
\mathscr{H}(I, \boldsymbol{v}) &=\mathscr{H}_{\mathrm{obs}}(I, \boldsymbol{v})+\alpha \mathscr{H}_{\mathrm{reg}}(\boldsymbol{v}) \\
&=\iint_{\Omega} f_{1}\left[\nabla I(\boldsymbol{x}, t) \cdot \boldsymbol{v}(\boldsymbol{x}, t)+\frac{\partial I(\boldsymbol{x}, t)}{\partial t}\right] \mathrm{d} \boldsymbol{x} \\
&+\alpha \iint_{\Omega} f_{2}[|\nabla u(\boldsymbol{x}, t)|+|\nabla v(\boldsymbol{x}, t)|]
\end{aligned}
\end{equation}
$$

### Large Displacement 修正 $\to$ Successive Linearization Within Multi-resolution

以上 optical flow 不論是 PDE 形式或是積分形式都是假設無限小的時間和空間的變化。實務上的時間和空間變化量不一定會很小，特別是 optical flow 物體移動很快，或是 video frame rate 太慢的時候。optical flow 算法 based on PDE or 積分都會有不準確的問題。因此我們需要考慮大位移的情況。此時 brightness conservation 如下。

Instead of $\eqref{OpticFlow}$, we have

$$
\begin{equation}
I( \mathbf{x}+ \mathbf{d(x)}, t+\Delta t) - I(\mathbf{x}, t) \approx 0 \label{OpticFlowD}
\end{equation}
$$

where 

* $\Delta t$ is the temporal sampling rate
* $\mathbf{d(x)}$ is the displacement from $t$ to $t+\Delta t$ of the point located at postion $\mathbf{x}$ at time $t$					

當 $\Delta t$ 和 $\mathbf{d(x)}$ 比較大時，$\eqref{OpticFlowD}$ 可能高度非線性。因此需要更厲害的方法，**基本所有的研究方向 (包含 AI 的方式) 都訴諸於 a succession of linearizations embedded within a multi-resolution scheme.** [@corpettiDenseEstimation2002]



Q: 為什麼需要 multi-resolution?  

A (my intuition): 2D optical flow 缺乏 depth information, 因此物體運動在 depth 的移動會造成其大小的改變。因此 multi-resolution 就變得非常重要。 



#### Succession of Linearization within Multi-Resolution

具體執行方式：Given a previous estimate $\tilde{\mathbf{d}}$ of the displacemtn field at a coarser resolution, a first order expansion of the first term in $\eqref{OpticFlowD}$ is performed around $(\mathbf{x}+ \tilde{\mathbf{d}}(\mathbf{x}), t+\Delta t)$，見下圖，會得到以下的 cost function

$$
\begin{equation}
\int_{\Omega} f_{1}\left[\nabla I(x+\tilde{d}(x), t+\Delta t) \cdot \Delta d(x)+I(x+\tilde{d}(x), \\
t+\Delta t)-I(x, t)\right]+\alpha f_{2}\left[|\nabla(\tilde{d}+\Delta d)(x)|\right] \label{Cost2}
\end{equation}
$$

to be minimized with respect to the displacement increment $\Delta d=d-\tilde{d}$ (Fig. 2).  BTW, 此處的 $\mathbf{d}$ 取代 $\boldsymbol{v}$ 的作用。

<img src="/media/image-20220131004558939.png" alt="image-20220131004558939" style="zoom:50%;" />

$\eqref{Cost2}$ 式基本假設 (i) brightness constancy; (ii) first-order smoothness.   **我們接下來修正這兩個假設。**



#### Brightness Constancy 修正 $\to$ Compressible Fluid Continuity Equation

Constant brightness 適用於剛性物體的運動亮度不變。但有例外 (i) 大小會改變的 motion, e.g. 有 depth 方向的移動；(ii) deformable motion, e.g. 氣體，心臟跳動。

修正的方法基於 continuity equation.  先看 fluid continuity equation (conservation of mass)

$$
\begin{equation}
\frac{\partial \rho}{\partial t}+\nabla \cdot(\rho \mathbf{V})=0 \label{FluidCont}
\end{equation}
$$

where

* $\rho$ is the fluid density
* $\mathbf{V}$ is the 3D velocity field

我們可以直接利用 $\eqref{FluidCont}$ 類比成 2D 的 image brightness $I$ and velocity $\boldsymbol{v}$ Satisfy:

$$
\begin{equation}
\frac{\partial I}{\partial t}+\nabla \cdot(I \boldsymbol{v})=0 \label{OpticCont}
\end{equation}
$$

$\nabla \cdot(I \boldsymbol{v}) = \nabla I \cdot \boldsymbol{v} + I (\nabla \cdot \boldsymbol{v})$.  對於 incompressible flow, i.e $\nabla \cdot \boldsymbol{v}=0$, $\eqref{OpticCont}$ 化簡為一般 optical flow equation $\eqref{OpticFlow}$.  相反對於 compressible flow, $\eqref{OpticCont}$ 比 optical flow equation $\eqref{OpticFlow}$ 多了一項 $I (\nabla\cdot \boldsymbol{v})$.  也就是換成全微分是要加上這多出的一項。

$$
\begin{equation}
\frac{d I}{d t}+ I \nabla \cdot \boldsymbol{v}=0 \label{OpticCont2}
\end{equation}
$$

$\eqref{Cost2}$ 也可以對應修正，基本要乘上一個 $\exp (\nabla\cdot \tilde d) $ term, 參考 [@corpettiDenseEstimation2002].  因為修正主要是對 deformable 移動，i.e. $\nabla \cdot \boldsymbol{v}\ne 0$，此處不再討論細節。



#### First Order Smoothness 修正 $\to$ Second Order Smoothness

上式 $f_2$ 是 1st order smoothness.

$$
\begin{equation}
\int_{\Omega}|\nabla u|^{2}+|\nabla v|^{2}
\end{equation}
$$

the same as those associated with order div-curl regularizer [41]

$$
\begin{equation}
\int_{\Omega} (\nabla \cdot d)^2 +(\nabla \times d)^2
\end{equation}
$$

where $\nabla \cdot \boldsymbol{d}=\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}$ and  $\nabla \times \mathbf{d} =\frac{\partial v}{\partial x}-\frac{\partial u}{\partial y}$

修正為 2nd order smoothness 為 gradient of div-curl regularizer. 

$$
\begin{equation}
\int_{\Omega} |\nabla (\nabla \cdot d)|^2 + |\nabla (\nabla \times d)|^2
\end{equation}
$$

Again, 本文主要是討論 AI optical flow,  細節可以參考 [@corpettiDenseEstimation2002].



## Neural Network Optical Flow

回到 neural network optical flow， 算法的基礎是 end-to-end training of $\eqref{OpticFlowD}$:  

* Input 是兩張相鄰 $\Delta t$ 的 frames:  $I( \mathbf{x}+ \mathbf{d(x)}, t+\Delta t)$ 和 $I(\mathbf{x}, t)$. 
* Output 是 $\mathbf{d(x)}$ 對每個 pixel $\mathbf{x}$.
* 我們使用一個 neural network to model 這個複雜的 input and output function.
* Training 的 cost/loss function 是每個像素的预测的光流（2维向量）和 ground truth 之间的欧式距离，这种误差为 EPE (End-Point-Error)，如下圖右上。


<figure>

<img src="/media/image-20220203003855492.png" alt="image-20220203003855492" style="zoom:80%;" />

<figcaption align="center"><b>Fig.1 - Optical Flow Neural Network. </b></figcaption>
</figure>


<figure>

<img src="/media/image-20220203011531620.png" alt="image-20220203011531620" style="zoom:67%;" />

<figcaption align="center"><b>Fig.2 - FlowNetC 堆疊 FlowNetS </b></figcaption>
</figure>



<figure>

<img src="/media/image-20220203010849039.png" alt="image-20220203010849039" style="zoom:100%;" />

<figcaption align="center"><b>Fig.3 - Schematic view of FlowNet2.0.  To compute large displacement optical flow we combine multiple FloeNets.   大括號代表 inputs concatenation.  Brightness Error is the difference between the 1st image and the 2nd image warped with the previously estimated flow.  To optimally deal with small displacements, we introduce smaller strides in the beginning and convolutions between upconvolutions into the FlowNetS architecture.  Finally we apply a small fusion network to provide the final estimate. </b></figcaption>
</figure>





<img src="/media/image-20220105224042468.png" alt="image-20220105224042468" style="zoom: 67%;" />



#### CORRESPONDENCES BETWEEN OPTICAL FLOW CNNS AND VARIATIONAL METHODS

We first provide a brief review for estimating optical flow using variational methods. In the next two sub-sections, we will bridge the correspondences between optical flow CNNs and classical variational methods.

The flow field can be estimated by minimizing an energy functional E of the general form



Search and Match problem



 to avoid overfitting.

## 
