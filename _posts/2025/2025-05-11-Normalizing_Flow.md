---
title: Math AI - Normalizing Flow
date: 2025-05-05 23:10:08
typora-root-url: ../../allenlu2009.github.io
categories:
  - GenAI
tags:
  - Fisher
  - Flow-Model
  - GenAI
  - Information
  - Math-AI
  - Math_AI
  - Score
---






## Normalizing Flow (NF) -  (自己到自己) 坐標變換 + Jacobian

**Normalizing: Jacobian matrix：為了 pdf normalize to 1.  這和 partition function 一個意思。**
**Flow:  invertible and differentiable：為了引入 neural network for maximum likelihood.**

設 $\phi: \mathbb{R}^d\to \mathbb{R}^d$ 為一個連續可微的函數，它將 $\mathbb{R}^d$ 中的元素進行變換，且其反函數 $\phi^{-1}: \mathbb{R}^d\to \mathbb{R}^d$ 也為連續可微。設 $p_0(x)$ 為 $\mathbb{R}^d$ 上的機率密度函數，而 $p_1(\cdot)$ 是由以下取樣程序所導出的密度：
$$\begin{aligned}
x \sim p_0\\
y = \phi(x),
\end{aligned}$$
對應於透過映射 $\phi$ 將 $p_0$ 的樣本進行轉換。我們可以使用變數變換計算 $p_1$ 的密度：

$$\begin{aligned}
p_1(y) &= p_0(\phi^{-1}(y)) \cdot \left| \det \left( \frac{\partial \phi^{-1}(y)}{\partial y} \right) \right|\\
&= \frac{p_0(x)}{\left| \det \left( \frac{\partial \phi(x)}{\partial x} \right) \right|}\quad\text{ with }x=\phi^{-1}(y)\\
\end{aligned}$$

最後的等號可以從 $\phi\circ\phi^{-1}(x) = \phi(\phi^{-1}(x)) = x$ 這一事實及鏈式法則的簡單應用中推導出來。這裡的 $\frac{\partial \phi^{-1}}{\partial x}$ 是反映射的雅可比矩陣（Jacobian matrix），大小為 $d \times d$，其元素為偏導數。

根據任務的不同——如是為了機率估計或是為了取樣——會根據情況選擇在 $\phi$ 或 $\phi^{-1}$ 空間中的表達方式。

Normalizing Flow 的觀念：將基礎分布 $p_0$ 經由已知變換 $\phi$ 轉換成另一個分布 $p_1$ 簡單，也用在一些產生比較複雜分佈的應用，例如從高斯分佈轉換成 Gamma 分佈。但直接應用在在生成模型（generative modeling）中會受到限制。需要透過某個變換 $\phi$ 將「簡單」的分布（如高斯分布）映射到近似資料分布。然而，但簡單的線性或非線性變換並不足以處理資料分布高度非高斯的情況。

這就引出了使用神經網絡作為靈活變換 $\phi_{\theta}(x)$ 的想法。關鍵任務便是最佳化這個神經網絡的參數 $\theta$。

### 透過 maximum likelihood 學習 flow 的參數

我們用 $p_\theta$ 來表示由 flow $\phi_\theta$ 所導出的參數化密度函數。

一個自然的最佳化目標是：最大化資料在模型下的機率（即最大似然）：

$$
arg \max_\theta E_{x\sim D}[ \log p_1(x)]
$$
將 $\phi_\theta$ 參數化為深度神經網絡時，會面臨幾個限制與挑戰，例如如何保證 $\phi_\theta$ 可微且可逆。以及如何有效計算 Jacobian!

一般比較有效率的計算：
1. Low rank approximation: not invertible
2. Autoregressive: sequential, performance bottleneck

所以需要對 flow 的架構加以限制，才有機會有效率。這是另一個關鍵，就是 break and conquer, 變成一次改一點，積少成多。用時間換取空間。這和 auto-regressive 有點類似？


## Full Rank Residual Flow - 一連串的坐標變換變成 Flow

一個 trade-off 就是 residual flow 架構，使用多個簡單的 residue 串聯一起：
![[Pasted image 20250509111804.png]]
![[Pasted image 20250509111954.png]]

接下來就是 make it continuous in time!


## 兩個視角：1. (自己到自己) 坐標變換；2. 運動方程

1. 從坐標變換的角度: $x_t \triangleq \phi_t\left(x_0\right)$, 並且 $x_0 = \phi_0\left(x_0\right)$.  $\phi_t$ 其實就是一連串的坐標變換。

2. 對於運動方程，也有兩種角度：一個是我不動，也就是同一個坐標 $x_t$ 隨時間的變化，得到第一個運動方程式: 速度場 = 位置的變化。
$$\frac{\mathrm{d} x_t}{\mathrm{~d} t}=u_t\left(x_t\right)$$ $$ x_t =x_0+\int_0^t u_s\left(x_s\right) \mathrm{d} s $$
另一個角度是：坐標變換後的運動方程式和原來的完全相同：同一個 vector field 速度場 = 坐標變換後位置的變化。
坐標變換:  $x_t \triangleq \phi_t\left(x_0\right)$

$$\frac{d \phi_t(x_0)}{d t}=u_t\left(\phi_t\left(x_0\right)\right)$$

這對 $\phi_t$ 是一個很強的限制：continuously differentiable, 並且 invertable (diffeomorphism)。因爲要 invertable, 所以 $x_t = \phi_t(x_0)$ 對於不同 initial conditional 的 $x_0$ 的“**流綫**”不能相交。因爲一旦相交就無法 invertable.  這和流體力學的 flow 基本一樣！

但是又有足夠的彈性，因爲是可壓縮的 flow, 也就是密度 (pdf) 是可以改變的，只要滿足 continuity equation 即可。


**流（flow）** 與 **向量場（vector field）** 的關係在 **normalizing flows** 和 **連續正規化（continuous normalizing flows, CNFs）** 中非常核心。讓我們一步一步拆解：

---

## 📦 對應到 Normalizing Flow

* 流 $\phi_t$ 就是學習的變換？**應該是學習向量場**？流是向量場的積分。
* 向量場 $u_t(x)$ 就是神經網路在不同時間產生的「方向指引」。
* 我們的目標就是要找到一組 $u_t$，使得經過流之後，基礎分佈 $p_0(x_0)$ 變成資料分佈 $p_1(x_1)$。
* 為了保證我們可以在生成與密度估計之間自由切換，這整個變換必須是流（invertible, smooth, no crossings）。

---

## Flow 的密度 $p_t$ 變化 from 坐標變換 Jacobian

$\phi_0$ 是流的初始條件，通常我們定義 $\phi_0(x_0) = x_0$，也就是初始時間 $t = 0$ 時的恆等映射（identity map）。因此，如果我們有一個隨機變數 $x_0 \sim p_0(x_0)$，那麼這就是流的起始密度。

---

### 🔁 經過 $\phi_t$ 變換後的密度：$p_t$

當這個樣本 $x_0$ 隨時間流動變成 $x_t = \phi_t(x_0)$，它的密度會隨之改變。根據**變數變換公式（change-of-variables formula）**，流經 $\phi_t$ 後的密度 $p_t(x_t)$ 與起始密度 $p_0(x_0)$ 的關係為：

$$
p_t(x_t) = p_0(x_0) \cdot \left| \det \left( \frac{\partial \phi_t^{-1}(x_t)}{\partial x_t} \right) \right|
$$

這個式子表示：密度會被映射的**體積變化率（Jacobian determinant）所調整。

也可以等價地寫成：

$$
\log p_t(x_t) = \log p_0(x_0) - \log \left| \det \left( \frac{\partial \phi_t(x_0)}{\partial x_0} \right) \right|
$$

## Flow 的密度 $p_t$ 變化 from 運動方程 (Continuity Equation)


## 🔁 背景設定 recap：

* 質點位置：$x_t = \phi_t(x_0)$，是從初始點 $x_0$ 隨時間由 flow 變換而得。
* 速度場（vector field）：$\frac{d x_t}{d t} = u_t(x_t)$
* 目標：表示密度 $p_t(x_t)$ 隨時間變化的關係。

---

## 📘 密度隨時間的變化公式（Continuity equation）

在連續流的語言中，密度的時間導數滿足：

$$
\frac{\partial}{\partial t} p_t(x_t) = - \nabla \cdot \left( p_t(x_t) \cdot u_t(x_t) \right)
$$

這稱為**連續性方程式**（continuity equation），源自質量守恆概念。**這裏是運用機率守恆。**

---

## 📌 展開右側：

$$
\frac{\partial}{\partial t} p_t(x_t) = - \left[ \nabla p_t(x_t) \cdot u_t(x_t) + p_t(x_t) \cdot \nabla \cdot u_t(x_t) \right]
$$

但我們更常看的，是對 **log density** 的變化，非常神奇從偏微分方程變成全微分方程還簡化。也就是：

---

## ✅ 用 $u_t(x_t)$ 表示的 log-density 變化公式：

$$
\frac{d}{dt} \log p_t(x_t) = - \nabla \cdot u_t(x_t)
$$

這是在流模型中經常使用的核心公式，因為它簡單地表示了密度如何隨著「空間的壓縮/膨脹」而變化。這是 Normalizing Flow 中用來追蹤密度演化的關鍵。推導參考 Appendix C.

---

### ✅ 說明

* 如果 $u_t$ 是 divergence-free（$\nabla \cdot u_t = 0$），那麼密度不變（體積守恆）。
* 如果 $\nabla \cdot u_t < 0$，表示 flow 正在壓縮空間 → 密度增加。
* 如果 $\nabla \cdot u_t > 0$，表示空間在擴張 → 密度下降。


## Flow 的密度 $p_t$ 變化一致性

Flow 密度從 continuity equation 的結果：
$$
\frac{d}{dt} \log p_t(x_t) = - \nabla \cdot u_t(x_t)
$$

和從 **flow coordinate change（坐標變換）** 的角度所得到的 density 變化公式如何連結？

$$
\log p_t(x_t) = \log p_0(x_0) - \log \left| \det \left( \frac{\partial \phi_t(x_0)}{\partial x_0} \right) \right|
$$


## 📌 關鍵觀察：Jacobian determinant 的時間導數

定義：

$$
J_t := \frac{\partial \phi_t(x_0)}{\partial x_0} \quad \text{（Jacobian）}
$$

一個重要事實是：in Appendix D

$$
\frac{d}{dt} \log \det J_t = \nabla \cdot u_t(x_t)
$$

所以我們從 flow 的 Jacobian 變化率，會得到：

$$
\frac{d}{dt} \log p_t(x_t) = - \frac{d}{dt} \log \det J_t = - \nabla \cdot u_t(x_t)
$$

這正是 **continuity equation** 給出的結果。

---

## ✅ 結論：兩者完全一致！

* Continuity equation 是從 **守恆律 + 對 Eulerian frame（固定坐標）** 的觀點出發。
* 坐標變換則是從 **Lagrangian frame（跟著粒子走）** 出發，利用 Jacobian 的變化追蹤密度的變化。
* 最終，兩者都導出：

  $$
  \boxed{
  \frac{d}{dt} \log p_t(x_t) = - \nabla \cdot u_t(x_t)
  }
  $$

## Flow Matching Paper

Beginner: https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html
Paper: FLOW MATCHING FOR GENERATIVE MODELING https://arxiv.org/pdf/2210.02747

![[Pasted image 20250508093906.png]]
這裡 $t \in [0, 1]$.   $p_t(x)$ 是 time-dependent probability, 就是 Fokker-Planck 的主角。但是 flow matching 是 deterministic?  還是有機率分佈？ 應該是 flow 在 t=0 有一個 distribution, 但是在移動的過程是 deterministic!  用 DDIM 思考比較容易。

對於 $t=0$,  對應一個機率分佈, say $p_0(x) = N(0, I)$.   我們任意 sample 一點 $x=X$.  
$\phi_t(X)$ 就是隨時間而且可微分而且可逆的 mapping，和 $v_t$ 向量場作用下的 **“position trajectory, or flow!”**

但是 fl. w matching 的重點是 vector field!  所以要建構一個類似 Fokker-Planck 的 equation, 就是 (1).  $\phi_t(x)$ 比較像一個 particle moving in the space.

- 機率分佈：$p_t(x)$ output 是正純量。
- Flow: $\phi_t(x)$ output 是 position, 從 position to position.
- Vector field: $v_t(x)$ output 而且 dimension 和 input $x$ 相同。從 position to velocity.

用下圖為例。最左邊的 distribution 是 $p_0(x)=N(0, I)$,  最右邊的 distribution 是 $p_t(x)$ 隨著時間變化。類似 Fokker-Planck equation 的 $p(t, x)$.
$\phi_t(x)$ 則是不同顏色的 trace or trajectory over time.  這裡沒有大小的觀念。基本是一條一條的不交叉的 deterministic flows.  $v_t(x)$ 則是 vector field 作用在 $\phi_t(x)$ 決定 flow 的流向。就是上式 (1) 的微分方程。


![[flow.gif]]


| Concept                                             | Analogy                                                            | Note                                     |
| --------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------- |
| $x_t \in \mathbb{R}^d$                              | Position of a charged particle at $t$                              |                                          |
| $u_t(x) \in \mathbb{R}^d \to \mathbb{R}^d$          | Vector/electrical field at time $t$.                               | NF 的關鍵                                   |
| $\phi_t(x) \in \mathbb{R}^d \to \mathbb{R}^d$       | Position of the particle at time $t$ after moving under the field  | NF 核心:Invertable and differentiable flow |
| $p_t(x) \in \mathbb{R}^d \to \mathbb{R}^+$          | Probability distribution                                           | NF Jacobian                              |
| $\frac{d}{dt} \phi_t(x) = u_t(\phi_t(x))$           | Newton’s law (no mass/inertia): particle velocity = field strength |                                          |
| $\frac{d}{dt} x_t = u_t$                            |                                                                    |                                          |
| $\frac{d}{dt} \log p_t(x_t) = \nabla\cdot u_t(x_t)$ |                                                                    |                                          |
| 另一個方法是下面 eq(4)                                      |                                                                    |                                          |

一旦有 $p_0(x)$ and $\phi_t(x)$ 可以得到 $p_t(x)$, 這對計算 likelihood 很有用。

![[Pasted image 20250508155947.png]]




## Reference

MIT 6.S184: Flow Matching and Diffusion Models https://www.youtube.com/watch?v=GCoP2w-Cqtg&t=28s&ab_channel=PeterHolderrieth

Yaron Meta paper: [[2210.02747] Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

An Introduction to Flow Matching: https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html

## Appendix A: Junk





![[Pasted image 20250510231300.png]]




**Diffusion SDE 的角度:
$$
d \boldsymbol{x}(t)=\boldsymbol{f}(\boldsymbol{x}(t), t) d t+g(t) d \boldsymbol{w}_t
$$

$g(t) = \sigma(t)$ 就是 brownian motion,  $\boldsymbol{f}(\boldsymbol{x}(t), t)$ 是 drift term, 但不等價於 flow.   但是最簡單的 $\boldsymbol{f}(\boldsymbol{x}(t), t) = -\theta \,\boldsymbol{x}_t$  可以轉換成 flow.  



第一個 ODE：是 **sample／trajectory ODE**，不是 distribution ODE！

$$
\frac{d x_t}{d t}=f\left(x_t, t\right)-\frac{1}{2} g^2(t) \nabla \log p\left(x_t, t\right),
$$

第二個 ODE 才是 **distribution ODE** (Lai，Chieh－Hsin paper：FP－Diffusion)
The log-likelihood can be exactly calculated by numerically solving the concatenated ODEs backward from $T$ to 0 ，after initialization with $\tilde{\boldsymbol{x}}_{\boldsymbol{\theta}}(0) \sim q_0(\boldsymbol{x})$

兩者可以合并如下：

$$
\begin{aligned}
& \frac{d}{d t}\left[\begin{array}{c}
\tilde{\boldsymbol{x}}_{\boldsymbol{\theta}}(t) \\
\log p_{t, \boldsymbol{\theta}}\left(\tilde{\boldsymbol{x}}_{\boldsymbol{\theta}}(t)\right)
\end{array}\right] = {\left[\begin{array}{c}
\boldsymbol{f}\left(\tilde{\boldsymbol{x}}_{\boldsymbol{\theta}}(t), t\right)-\frac{1}{2} g^2(t) \boldsymbol{s}_{\boldsymbol{\theta}}\left(\tilde{\boldsymbol{x}}_{\boldsymbol{\theta}}(t), t\right) \\
\frac{1}{2} g^2(t) \nabla_{\boldsymbol{x}}\cdot\left(\boldsymbol{s}_{\boldsymbol{\theta}}\left(\tilde{\boldsymbol{x}}_{\boldsymbol{\theta}}(t), t\right)\right)-\nabla_{\boldsymbol{x}}\cdot\left(\boldsymbol{f}\left(\tilde{\boldsymbol{x}}_{\boldsymbol{\theta}}(t), t\right)\right)
\end{array}\right] . }
\end{aligned}
$$




![[Pasted image 20250510220354.png]]

第一個 ODE: 是 sample/trajectory ODE, 不是 distribution ODE!
![[Pasted image 20250510223159.png]]
第二個 ODE 才是 distribution ODE (Lai, Chieh-Hsin paper: FP-Diffusion).

兩者可以合并如下:
![[Pasted image 20250510220343.png]]



從 Flow ODE 的角度：([An introduction to Flow Matching · Cambridge MLG Blog](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html))

![[Pasted image 20250510231300.png]]







$$
\begin{aligned}
& \frac{\mathrm{d} x_t}{\mathrm{~d} t}=\lim _{\delta \rightarrow 0} \frac{x_{t+\delta}-x_t}{\delta}=\frac{\phi_t\left(x_t\right)-x_t}{\delta}=u_t\left(x_t\right) \\
& x_t \triangleq \phi_t\left(x_0\right)=x_0+\int_0^t u_s\left(x_s\right) \mathrm{d} s \\
& \frac{d \phi_t}{d t}=u_t\left(\phi_t\left(x_0\right)\right)
\end{aligned}
$$


#### Continuous Time Limit
![[Pasted image 20250509112059.png]]

所以 $\phi_t$ 和 $u_t$ 都是 deterministic.   但是 initial $x_0$  是 stochastic, 因此 $x_t$ 也是 stochastic.  對應 $p_t(x_t)$  distribution.

Question:  如何用這樣的 formulation 解釋 ODE 的 DDIM?
Question:  如何用這樣的 formulation 解釋 SDE 的 VE (類似 NCSN), VP (類似 DDPM)?
- 可以偷換觀念？把 ODE 再解釋成 SDE?

### 還有 Jacobian

剛才只是 $\phi_t$ 部分，還有 Jacobian 部分。也就是 normalization 的部分。對應就是 $\int p_t(x) dx = 1$
從流體力學的角度，就是隨時間變化 = divergence of density!  在電磁學稱為 continuity equation.  在 CNF 稱為 transport equation!

![[Pasted image 20250509130907.png]]
$\mathbf{j}=\rho \mathbf{v}$,  無源的公式很簡單。


更普遍流體力學的 transport equation 或是 convection-diffusion equation 如下：
![[Pasted image 20250509130208.png]]

上式的 $\Phi$ 對應需要保守的量 $p_t$；沒有 diffusion 和 source. 所以簡單不少。

![[Pasted image 20250509115439.png]]


![[Pasted image 20250509130805.png]]

以下部分是從 paper 來的，重複。

## Appendix B:

太好了！我們現在來推導這個重要公式：

$$
\boxed{
\frac{d}{dt} \log \det J_t = \nabla \cdot u_t(x_t)
}
$$

其中：

* $J_t = \frac{\partial \phi_t(x_0)}{\partial x_0}$ 是 flow $\phi_t$ 的 Jacobian matrix。
* $u_t(x_t) = \frac{d}{dt} \phi_t(x_0)$ 是 flow 的速度場。
* $\nabla \cdot u_t(x_t) = \mathrm{tr}\left(\frac{\partial u_t(x_t)}{\partial x_t}\right)$ 是速度場的 divergence。

---

## 🧭 目標

從 $J_t = \frac{\partial \phi_t}{\partial x_0}$ 推導出：

$$
\frac{d}{dt} \log \det J_t = \mathrm{tr}\left( J_t^{-1} \cdot \frac{d J_t}{dt} \right)
$$

然後連結到 divergence。

---

## 🧮 步驟一：矩陣行列式的微分公式

對於一個可微矩陣 $A(t)$，有一個經典公式：

$$
\frac{d}{dt} \log \det A(t) = \mathrm{tr}\left( A^{-1}(t) \cdot \frac{dA(t)}{dt} \right)
$$

應用到我們的情況：

$$
\frac{d}{dt} \log \det J_t = \mathrm{tr}\left( J_t^{-1} \cdot \frac{dJ_t}{dt} \right)
$$

---

## 🧮 步驟二：計算 $\frac{d}{dt} J_t$

由於：

$$
J_t = \frac{\partial \phi_t(x_0)}{\partial x_0}
$$

而 $\frac{d}{dt} \phi_t(x_0) = u_t(\phi_t(x_0))$

用鏈式法則對 $J_t$ 微分：

$$
\frac{d}{dt} J_t = \frac{\partial}{\partial x_0} \left[ \frac{d}{dt} \phi_t(x_0) \right]
= \frac{\partial}{\partial x_0} u_t(\phi_t(x_0))
= \frac{\partial u_t}{\partial x_t} \cdot \frac{\partial \phi_t(x_0)}{\partial x_0}
= \left( \nabla_{x_t} u_t \right) \cdot J_t
$$

---

## 🧮 步驟三：代入上式

帶入：

$$
\frac{d}{dt} \log \det J_t
= \mathrm{tr}\left( J_t^{-1} \cdot \left( \nabla_{x_t} u_t \cdot J_t \right) \right)
= \mathrm{tr}\left( \nabla_{x_t} u_t \right)
= \nabla \cdot u_t(x_t)
$$

---

## ✅ 結論

我們完成推導了：

$$
\boxed{
\frac{d}{dt} \log \det \left( \frac{\partial \phi_t}{\partial x_0} \right)
= \nabla \cdot u_t(x_t)
}
$$

這個關係連結了 flow 的 Jacobian determinant 和速度場的 divergence，並在 normalizing flows 中扮演核心角色！

---

要我示範怎麼在一維或二維情況下具體算這個嗎？


## Appendix C:

從偏微分 **continuity equation**（連續性方程式）出發，逐步推導出 log-density 的變化公式：

---

## 📌 起點：Continuity equation

連續性方程描述在速度場 $u_t(x)$ 下，**密度 $p_t(x)$** 如何隨時間變化：

$$
\frac{\partial p_t(x)}{\partial t} + \nabla \cdot (p_t(x) \cdot u_t(x)) = 0
$$

這是質量守恆的數學形式。

---

## 🔁 展開 divergence 項

使用 product rule（乘法微分）展開右邊：

$$
\frac{\partial p_t(x)}{\partial t} + u_t(x) \cdot \nabla p_t(x) + p_t(x) \cdot \nabla \cdot u_t(x) = 0
$$

---

## 🔄 重組項目

把左邊的項移到右邊，整理為：

$$
\frac{\partial p_t(x)}{\partial t} = - u_t(x) \cdot \nabla p_t(x) - p_t(x) \cdot \nabla \cdot u_t(x)
$$

---

## 🎯 目標：log-density 的時間導數

我們的目標是計算 $\frac{d}{dt} \log p_t(x_t)$，也就是追蹤沿著粒子路徑 $x_t$ 的密度變化。這需要用**全導數**（total derivative），因為 $x_t$ 本身也在變動。

---

### ✅ 使用鏈式法則

記住：

$$
\frac{d}{dt} \log p_t(x_t) = \frac{\partial}{\partial t} \log p_t(x) \Big|_{x = x_t} + \frac{dx_t}{dt} \cdot \nabla \log p_t(x) \Big|_{x = x_t}
$$

利用鏈式法則與前面展開式：

$$
\frac{d}{dt} \log p_t(x_t)
= \frac{1}{p_t(x_t)} \left[ \frac{\partial p_t(x_t)}{\partial t} + \frac{d x_t}{dt} \cdot \nabla p_t(x_t) \right]
$$

---

## 🔄 代入 continuity equation 的結果

$$
\frac{\partial p_t(x)}{\partial t} + u_t(x) \cdot \nabla p_t(x) = -p_t(x) \cdot \nabla \cdot u_t(x)
$$

帶入上式得到：

$$
\frac{d}{dt} \log p_t(x_t) = \frac{1}{p_t(x_t)} \left[ -p_t(x_t) \cdot \nabla \cdot u_t(x_t) \right]
= - \nabla \cdot u_t(x_t)
$$

---

## ✅ 結論

$$
\boxed{
\frac{d}{dt} \log p_t(x_t) = - \nabla \cdot u_t(x_t)
}
$$

這就是我們想要的 **log-density 演化方程**，它告訴我們密度的 log 值如何隨著 flow 被壓縮或擴張而變化。


## Appdendix D:

推導這個重要公式：

$$
\boxed{
\frac{d}{dt} \log \det J_t = \nabla \cdot u_t(x_t)
}
$$

其中：

* $J_t = \frac{\partial \phi_t(x_0)}{\partial x_0}$ 是 flow $\phi_t$ 的 Jacobian matrix。
* $u_t(x_t) = \frac{d}{dt} \phi_t(x_0)$ 是 flow 的速度場。
* $\nabla \cdot u_t(x_t) = \mathrm{tr}\left(\frac{\partial u_t(x_t)}{\partial x_t}\right)$ 是速度場的 divergence。

---

## 🧭 目標

從 $J_t = \frac{\partial \phi_t}{\partial x_0}$ 推導出：

$$
\frac{d}{dt} \log \det J_t = \mathrm{tr}\left( J_t^{-1} \cdot \frac{d J_t}{dt} \right)
$$

然後連結到 divergence。

---

## 🧮 步驟一：矩陣行列式的微分公式

對於一個可微矩陣 $A(t)$，有一個經典公式：

$$
\frac{d}{dt} \log \det A(t) = \mathrm{tr}\left( A^{-1}(t) \cdot \frac{dA(t)}{dt} \right)
$$

應用到我們的情況：

$$
\frac{d}{dt} \log \det J_t = \mathrm{tr}\left( J_t^{-1} \cdot \frac{dJ_t}{dt} \right)
$$

---

## 🧮 步驟二：計算 $\frac{d}{dt} J_t$

由於：

$$
J_t = \frac{\partial \phi_t(x_0)}{\partial x_0}
$$

而 $\frac{d}{dt} \phi_t(x_0) = u_t(\phi_t(x_0))$

用鏈式法則對 $J_t$ 微分：

$$
\frac{d}{dt} J_t = \frac{\partial}{\partial x_0} \left[ \frac{d}{dt} \phi_t(x_0) \right]
= \frac{\partial}{\partial x_0} u_t(\phi_t(x_0))
= \frac{\partial u_t}{\partial x_t} \cdot \frac{\partial \phi_t(x_0)}{\partial x_0}
= \left( \nabla_{x_t} u_t \right) \cdot J_t
$$

---

## 🧮 步驟三：代入上式

帶入：

$$
\frac{d}{dt} \log \det J_t
= \mathrm{tr}\left( J_t^{-1} \cdot \left( \nabla_{x_t} u_t \cdot J_t \right) \right)
= \mathrm{tr}\left( \nabla_{x_t} u_t \right)
= \nabla \cdot u_t(x_t)
$$

---

## ✅ 結論

我們完成推導了：

$$
\boxed{
\frac{d}{dt} \log \det \left( \frac{\partial \phi_t}{\partial x_0} \right)
= \nabla \cdot u_t(x_t)
}
$$

這個關係連結了 flow 的 Jacobian determinant 和速度場的 divergence，並在 normalizing flows 中扮演核心角色！
