---
title: Math AI - Improve Flow Matching
date: 2025-06-02 23:10:08
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




https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html


![[Pasted image 20250602162706.png]]
## Takeaway

Flow matching 神奇三部曲如下：
$$
\begin{aligned}
&\mathcal{L}_{\mathrm{FM}}(\theta)=\mathbb{E}_{t, p_t(x)}\left\|v_t(x)-u_t(x)\right\|^2\\
&\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t, q(x_1), p_t(x \mid x_1)}\left\|v_t(x)-u_t(x \mid x_1)\right\|^2,\\
&\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t, q(x_1), p(x_0)}\left\|v_t(\psi_t(x_0))-\frac{d}{d t} \psi_t\left(x_0\right)\right\|^2
\end{aligned}
$$
Step1:  Global flow matching
Step2:  Conditional flow match: 注意
- $\mathcal{L}_{\mathrm{FM}} \ne \mathcal{L}_{\mathrm{CFM}}$，**但是** $\min\mathcal{L}_{\mathrm{FM}} \equiv \min\mathcal{L}_{\mathrm{CFM}}$
- 所以 $u_t(x) \ne u_t\left(x \mid x_1\right)$ => global flow 和 conditional flow 可能不一致？
Step3:  轉換成可以 sample 的 distribution, $t, x_1, x_0$
- **重點是如何假設 $\psi_t(x_0)$ 和 $x_0, x_1$ 的關係。最簡單就是 linear interpolation.**



**Diffusion process:** 
- **物理意義/現象是從頭到脚的 stochastic SDE**
- 經由對應的 Fokker-Planck ODE equation 可以得到每個時間的 $p_t(x)$, 以及 forward/reverse dynamics.   
- 其中 deterministic score function (vector field) $\nabla_{x} \log p_t(x)$ 扮演關鍵的角色。發動者是 forward path 加 noise; 關鍵是 reverse path score function 的引導者。
- **訓練 neural network 近似 score function:  score matching!**   
- Sampling 時則是利用 random sampling from initial distribution -> Langevin dynamics + denoiser (score function) 產生 samples
- Likelihood: 還是要經過 Fokker-Planc ODE 計算出.

**Flow process:** 
- **物理意義/現象是 deterministic ODE + random initial condition/distribution** 
- 物理順序是: vector field $u_t$ + initial condition 產生-> trajectory $X_t$ 產生-> flow $\phi_t(x_t)$ 產生-> distribution $p_t(x_t)$
- Flow equation 是非常簡單的 ODE:  $X_0 = x_0, \frac{d X_t}{dt} = u_t(X_t)$,  $X_t$ 是每一個 **trajectory** 對應不同的 initial condition $x_0$. 
- 把所有的 initial condition trajectories 集合起來就是 flow: $\phi_t(X_t)$，當然也滿足 $\frac{d \phi_t(X_t)}{dt} = u_t(\phi_t(X_t))$ 
- **訓練 neural network 近似 vector fild $u_t$ (是 flow 的微分):  flow matching!** 
- Sampling 時則是利用 **random sample initial distribution** ->  flow ODE 產生 samples
- Likelihood: 基本經過 reversed flow 可以計算出.


Flow matching is simulation-free (不用 ODE solver) in training! but need ODE solver in sampling!
Consistency mode is simulation-free in sampling!
    

### Diffusion Process 套用流體力學


**注意偏微分和全微分的差別**
$$\frac{d f(\mathbf{x},t)}{dt} = \frac{\partial f(\mathbf{x},t)}{\partial t} + \frac{d\mathbf{x}}{dt}\cdot \nabla f(\mathbf{x},t) = (\frac{\partial}{\partial t} + \frac{d\mathbf{x}}{dt}\cdot \nabla) f(\mathbf{x},t) =  (\frac{\partial}{\partial t} + \mathbf{u}\cdot \nabla) f(\mathbf{x},t)$$
**Continuity equation**
1. **Eulerian form (偏微分)**: $\partial_t p +\nabla\cdot(p\,u)=0.$
2. **Lagrangian form (全微分)**: $\displaystyle -\frac{d\log p}{dt}=-\bigl(\partial_t+u\cdot\nabla\bigr)\log p= \nabla\cdot u,$

Lagrangian form 很有意思把 Flow and Score function 連結一起：
$-\partial_t \log p = u\cdot\nabla\log p + \nabla\cdot u,$

對應的全微分：[FP-Diffusion: Improving Score-based Diffusion Models by Enforcing the Underlying Score Fokker-Planck Equation](https://arxiv.org/pdf/2210.04296)

$$
\begin{aligned}
\frac{d \log p_{t, \boldsymbol{\theta}}{(\boldsymbol{x}}_{\boldsymbol{\theta}}(t))}{d t} &= -\nabla_{\boldsymbol{x}}(\boldsymbol{f}({\boldsymbol{x}}_{\boldsymbol{\theta}}(t), t)) + \frac{1}{2} g^2(t) \nabla_{\boldsymbol{x}}(\boldsymbol{s}_{\boldsymbol{\theta}}({\boldsymbol{x}}_{\boldsymbol{\theta}}(t), t)) \\
\end{aligned}
$$

### Flow ODE 對應流體力學：
([An introduction to Flow Matching · Cambridge MLG Blog](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html))

$$
\begin{aligned}
& \frac{d}{d t}\left[\begin{array}{c}
\boldsymbol{x}_t \\
\log p_{t}\left(\boldsymbol{x}_t \right)
\end{array}\right] = {\left[\begin{array}{c}
\mathbf{u}_{\theta}(t, \boldsymbol{x}_t)\\
-\nabla_{\boldsymbol{x}}\cdot\mathbf{u}_{\theta}(t, \boldsymbol{x}_t)
\end{array}\right] . }
\end{aligned}
$$
Flow 另外多了一個 $\phi(\boldsymbol{x}_t)$ 就是 flow 的概念：$\frac{d}{d t} \phi(\boldsymbol{x}_t) \\= \mathbf{u}_{\theta}(t, \phi(\boldsymbol{x}_t))$
Why flow $\phi(\boldsymbol{x}_t)$?  滿足上式的 $u(t)$ 有無限多 from $N(0, I)$ to $p_{data}$!  需要加上 structure，這個 structure 就是$\phi(\boldsymbol{x}_t)$。

![[Pasted image 20250511011520.png]]

### Diffusion SDE  對應流體力學：
$$
d \boldsymbol{x}(t)=\boldsymbol{f}(\boldsymbol{x}(t), t) d t+g(t) d \boldsymbol{w}_t
$$
$\boldsymbol{x}(t)$ 是 SDE equation 的 trajectory，$\tilde{\boldsymbol{x}}(t)$ 則是對應的 deterministic ODE 的 trajectory. 
$$
\begin{aligned}
& \frac{d}{d t}\left[\begin{array}{c}
\tilde{\boldsymbol{x}}_{\boldsymbol{\theta}}(t) \\
\log p_{t, \boldsymbol{\theta}}\left(\tilde{\boldsymbol{x}}_{\boldsymbol{\theta}}(t)\right)
\end{array}\right] = {\left[\begin{array}{c}
\boldsymbol{f}\left(\tilde{\boldsymbol{x}}_{\boldsymbol{\theta}}(t), t\right)-\frac{1}{2} g^2(t) \boldsymbol{s}_{\boldsymbol{\theta}}\left(\tilde{\boldsymbol{x}}_{\boldsymbol{\theta}}(t), t\right) \\
\frac{1}{2} g^2(t) \nabla_{\boldsymbol{x}}\cdot\left(\boldsymbol{s}_{\boldsymbol{\theta}}\left(\tilde{\boldsymbol{x}}_{\boldsymbol{\theta}}(t), t\right)\right)-\nabla_{\boldsymbol{x}}\cdot\left(\boldsymbol{f}\left(\tilde{\boldsymbol{x}}_{\boldsymbol{\theta}}(t), t\right)\right)
\end{array}\right]\\
= \left[\begin{array}{c}
\mathbf{v}_{\theta}(t, \boldsymbol{x}_t)\\
-\nabla_{\boldsymbol{x}}\cdot\mathbf{v}_{\theta}(t, \boldsymbol{x}_t)
\end{array}\right]
. }
\end{aligned}
$$

where $$\mathbf{v}_{\theta}(t, \boldsymbol{x}_t) =\boldsymbol{f}\left(\tilde{\boldsymbol{x}}_{\boldsymbol{\theta}}(t), t\right)-\frac{1}{2} g^2(t) \boldsymbol{s}_{\boldsymbol{\theta}}\left(\tilde{\boldsymbol{x}}_{\boldsymbol{\theta}}(t), t\right) $$
### 相同和相異

1.  Diffusion SDE -> Diffusion ODE 的 $\boldsymbol{f} - \frac{1}{2} g^2(t) \boldsymbol{s}_{\theta}$  其實就等價於 Flow ODE 中的 $\boldsymbol{u}$!    但在 Diffusion $\boldsymbol{f}$ 是 pre-determined 簡單的 drift function (VP) 或是 0 (VE).  重點在 score function and score matching.
2. Flow ODE 是用 neural network 直接做 flow matching！
3. Flow 裏面沒有 score function 的角色，因爲 $g^2(t) = \frac{d\sigma^2(t)}{dt} = 2\sigma(t) \dot{\sigma}(t)=0$！因爲不需要燈塔的指引找到回家的路！或是回家的路已經合并在 $\boldsymbol{f}$ 這個 vector field 裏！所以 Flow 並不是 stochastic process，而是 deterministic process + random initial sampling.  

Question:  在 DDPM to DDIM,  使用同一個 score function, 但是設定 $\sigma =0$ 變成 DDIM.
是否此處也是可以讓 $\boldsymbol{f}' = \boldsymbol{f} - \frac{1}{2} g^2(t) \boldsymbol{s}_{\theta}$，直接用 flow 的方法得到 sample?  如此也可以比較 Flow, VE, VP?



## From Diffusion Process to Flow Framework

如同宋颺在 interview 所提到，當初在發展 Diffusion SDE (continuous model) 是被 DDPM (denoising diffusion probabilistic model, discrete model) 刺激，可以統一解釋 score matching and DDPM.   

他知道 diffusion SDE 可以轉換成 **ODE** using Fokker-Planck equation.  不過他的目的是計算 likelihood 可以比較不同的 diffusion methods performance (NLL - negative likelihood)。很快他發現 ODE 可以加速，因為 (1) ODE solver 本來執行速度就比 SDE solver 快；（2) 宋颺另外針對 diffusion 提出 predictor (score ?) and corrector (Langevin dynamics) 的方法。
Q: (2) 是 improve quality or speed?

ODE 另一個角度是從 Flow 出發。Flow 本身是完全 deterministics.  但是把 initial condition 視爲 random distribution.  

**Flow Matching = Continuous Normalizing Flows + Diffusion Models？**

Flow match 包含: Diffusion score matching,   OT (Optimal Transport)



## Flow Matching Papers


Beginner: https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html

Paper: FLOW MATCHING FOR GENERATIVE MODELING https://arxiv.org/pdf/2210.02747

![[Pasted image 20250508093906.png]]

以上 paper 的 notation 很難了解。我加上註解：
$t \in [0, 1]$，$p_t(x)$ 是 time-dependent probability 被 vector field $u_t$ 驅動。  一般 $t=0$ 是容易 sampling distribution, e.g. $p_0(x) \sim N(0, I)$;  $t=1$ 是 data distribution $p_1(x) \sim p_{data}(x) = q(x)$

### Flow $\phi_t$ = 連續的 (自己到自己) 座標變換
**Flow** $\phi_t$ 定義:  $x_t \triangleq \phi_t\left(x_0\right)$,  顯然 $t=0 \to x_0 = \phi_0\left(x_0\right)$    $\phi_t(x)$ 必須可微分且可逆。

(重要) 同時 $x_t$  也要滿足運動方程：$\frac{d x_t}{d t}=u_t(x_t)$
所以會得到以下的方程式：
$$\frac{d \phi_t(x_0)}{d t}=u_t\left(\phi_t\left(x_0\right)\right)$$
因為 $\phi_t(x)$ 是自己到自己的座標變換，在某些情況，可以省掉 subscript.

$$\frac{d \phi_t(x)}{d t}=u_t\left(\phi_t\left(x\right)\right)$$
雖然 $\phi_t$ 必須可微分且可逆，已經是很強的限制。但仍然有無窮多的 $\phi_t$ and $u_t$ 滿足 $p_0(x)$ 到 $p_1(x)$ 的分佈轉換。

我們可以從密度守恆再加上更嚴格的條件 (讓 flow 更 smooth? or minimum energy?)

### Flow 的密度 $p_t$ 守恆 from Jacobian

密度守恆表現在：（1）座標變換的 Jacobian, 如下文。
在上下文清楚的情況，可以省掉 $x$ 的下標 $0$ or $t$。

$$\begin{aligned}
p_t(x_t) &= p_0(x_0) \cdot \left| \det \left( \frac{\partial \phi_t^{-1}(x_t)}{\partial x_t} \right) \right| \\
p_t(x) &= p_0(\phi_t^{-1}(x)) \cdot \left| \det \left( \frac{\partial \phi_t^{-1}(x)}{\partial x} \right) \right|
\end{aligned}
$$

這個式子表示：密度會被映射的**體積變化率（Jacobian determinant）所調整。

也可以等價地寫成：

$$
\log p_t(x) = \log p_0(\phi_t^{-1}(x)) - \log \left| \det \left( \frac{\partial \phi_t(x)}{\partial x} \right) \right|
$$

### Flow 的密度 $p_t$ 守恆 from Continuity Equation

![[Pasted image 20250512113341.png]]

(2) 這個 $p_0(x)$ 到 $p_1(x)$ 的轉換同時要滿足 continuity equation.  因此 continuity equation 可以用來驗證或是限制 $u_t$.  (paper Appendix B 是錯的)

先是偏微分形式：![[Pasted image 20250513180714.png]]
可以推導出全微分形式更簡潔：
$$
\frac{d}{dt} \log p_t(x) = - \nabla \cdot u_t(x)
$$
積分形式（一般不會用積分來算 $p_t$ or likelihood，而是用座標變換偏微分形式，見下例 Ans3）
$$\log p_1(x_1)=\log p_0(x_0)−\int_0^1 \nabla \cdot u_t(x_t,t))dt$$

用下圖為例。最左邊的 distribution 是 $p_0(x)=N(0, I)$,  最右邊的 distribution 是 $p_t(x)$ 隨著時間變化 $t: 0\to 1$。類似 Fokker-Planck equation 的 $p(t, x)$.
$\phi_t(x)$ 則是不同顏色的 trace or trajectory over time.  這裡沒有大小的觀念。基本是一條一條的不交叉的 deterministic flows.  $v_t(x)$ 則是 vector field 作用在 $\phi_t(x)$ 決定 flow 的流向。就是上式 (1) 的運動方程。

![[flow.gif]]


## Flow Summary Table

| Concept                                                                      | Analogy                                                            | Note                               |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------- |
| $x_t \in \mathbb{R}^d$                                                       | Position of a charged particle at $t$                              | Particle 軌跡                        |
| $x_t= \phi_t(x_0) \in \mathbb{R}^d \to \mathbb{R}^d$ $x_0 = \phi_0(x_0)$     | Position of the particle at time $t$ from 座標變換。                    | Invertible and differentiable flow |
| $u_t(x) \in \mathbb{R}^d \to \mathbb{R}^d$                                   | Vector/electrical field at time $t$.                               | Flow match 關鍵                      |
| $p_t(x) \in \mathbb{R}^d \to \mathbb{R}^+$                                   | Probability distribution                                           | Flow 密度                            |
| $\frac{d}{dt} x_t = u_t(x_t)$, $\frac{d}{dt} \phi_t(x_0) = u_t(\phi_t(x_0))$ | Newton’s law (no mass/inertia): particle velocity = field strength | particle/flow 運動方程                 |
| $\frac{d}{dt} \log p_t(x_t) = \nabla\cdot u_t(x_t)$                          |                                                                    | Flow 密度守恆                          |
| 另一個方法是下面 eq(4)                                                               |                                                                    | Flow 密度守恆, Jacobian                |

一旦有 $p_0(x)$ and $\phi_t(x)$ 可以得到 $p_t(x)$, 這對計算 likelihood 很有用。

![[Pasted image 20250508155947.png]]

#### 我們看一個簡單的 flow (MIT Flow and Diffusion course)

**Example 1: Linear ODE**

Simple vector field:
$$
u_t(x)=-\theta x \quad(\theta>0)
$$
Claim: Flow is given by
$$
x_t = \psi_t\left(x_0\right)=\exp (-\theta t) x_0
$$
Proof:
1. Initial condition:
$$
\psi_t\left(x_0\right)=\exp (0) x_0=x_0
$$
1. ODE:
$$
\frac{\mathrm{d}}{d t} \psi_t\left(x_0\right)=\frac{\mathrm{d}}{d t}\left(\exp (-\theta t) x_0\right)=-\theta \exp (-\theta t) x_0=-\theta \psi_t\left(x_0\right)=u_t\left(\psi_t\left(x_0\right)\right)
$$
![[Pasted image 20250513093110.png]]

Q:  假設 $p_0(x) = p(x_0) = N(0, 1)$, what is $p_1(x)$?

**Ans1 直覺法:**  $x_1 = e^{-\theta} x_0$, 就是每條 trajectory 都被 scaled down by a factor $e^{-\theta}$
Key fact:  If $Z \sim \mathcal{N}(0, \sigma^2)$, then $aZ \sim \mathcal{N}(0, a^2 \sigma^2)$.
So:
$$
p_1(x) = p(x_1) = \mathcal{N}\left(0, e^{-2\theta}\right)
$$
**Ans2 Continuity equation 積分形式**:  推導在 Appendix C
$$\log p_1(x_1)=\log p_0(x_0)−\int_0^1 \nabla \cdot u_t(x_t,t))dt$$
$$
\nabla \cdot u_t(x_t, t) = \frac{d}{dx} (-\theta x_t) = -\theta
$$
$$
\int_0^1 \nabla \cdot u_t(x_t, t) \, dt = \int_0^1 (-\theta) \, dt = -\theta
$$
另外： $x_1 = e^{-\theta} x_0$  and $x_0 = e^{+\theta} x_1$
$$\begin{aligned}\log p_1(x_1)&=\log p_0(x_0)−\int_0^1 \nabla \cdot u_t(x_t,t))dt\\
&= \left(- \log \sqrt{2\pi}- \frac{x_0^2}{2}\right) +\theta= -\log \sqrt{2\pi e^{-2\theta}}- \frac{( x_1)^2}{2e^{-2\theta}}\\
&= \mathcal{N}(0, e^{-2\theta})
\end{aligned}$$

**Ans3 座標變換 偏微分形式**:  
$$
x_t = \psi_t\left(x_0\right)=\exp (-\theta t) x_0
$$
$$
\psi_t^{-1}\left(x\right)=\exp (+\theta t) x
$$
$$
\frac{\partial \psi_t^{-1}}{\partial x}\left(x\right)=\exp (+\theta t)
$$
$$\begin{aligned}
p_t(x) &= p_0(\phi_t^{-1}(x)) \cdot \left| \det \left( \frac{\partial \phi_t^{-1}(x)}{\partial x} \right) \right|\\
&= \frac{1}{\sqrt{2\pi}} e^{-\frac{x_0^2}{2}}\cdot e^{\theta t} = \frac{1}{\sqrt{2\pi}} e^{-\frac{(e^{\theta t}x_1)^2}{2}}\cdot e^{\theta t}\\
&= \frac{1}{\sqrt{2\pi e^{-2\theta t}}} e^{-\frac{x_0^2}{2e^{-2\theta t}}} = N(0, e^{-2\theta t})
\end{aligned}
$$

#### 結論：
- 簡單的例子都 ok, 但複雜的例子座標變換偏微分形式比較容易得到 $p_t(x)$ 隨時間變化。
- Flow/Vector field $u_t$ 和 probability distribution $p_t$ 原來是 decouple.  給定 Flow (or 座標變換) 可以轉換 $p_0$  to $p_1$.
- 在 Flow matching 剛好相反，是給定 $p_1$ 要找到 $u_t$.   實務上我們沒有 $p_1$, 只有 $p_1$ 產生的 samples.


**Example 2: Affine (linear + shift)** vector field:

$$
u_t(x) = -\theta x + b \quad (\theta > 0,\, b \in \mathbb{R}^d)
$$

### 🔁 ODE for the flow

We want to solve the differential equation:
$$
\frac{dx_t}{dt} = -\theta x_t + b
$$
The general solution is:

$$
x_t = (x_0 - \tfrac{b}{\theta}) e^{-\theta t} + \tfrac{b}{\theta}
$$
### 🧠 Interpretation

* As $t \to \infty$, the flow converges to the **fixed point** $\tfrac{b}{\theta}$ where $\frac{dx_t}{dt}=0$
* The system **contracts exponentially** toward this fixed point
* If $b = 0$, this reduces to the pure linear case with exponential decay to zero

## 🧮 1. **Jacobian of the flow**

Let’s define the flow map:

$$
\phi_t(x_0) = x_t = \left(x_0 - \frac{b}{\theta}\right) e^{-\theta t} + \frac{b}{\theta}
$$

The Jacobian of $x_t$ with respect to $x_0$ is:

$$
J_t(x_0) = \frac{\partial x_t}{\partial x_0} = e^{-\theta t} I
$$

where $I$ is the identity matrix.

---

## 📦 2. **Change in volume (determinant of Jacobian)**

This tells us how much a small volume of space shrinks or expands under the flow.

$$
\det(J_t) = e^{-d \theta t}
$$

where $d$ is the dimension of $x$, $d=1$ for 1D case. So the volume decays **exponentially** over time.

---

## 📉 3. **Change in density (under pushforward)**

If a particle’s density at time $t=0$ is $p_0(x_0)$, then the density at time $t$ is:

$$
p_t(x_t) = p_0(x_0) \cdot \left| \det\left( \frac{\partial x_t}{\partial x_0} \right)^{-1} \right| = p_0(x_0) \cdot e^{d \theta t}
$$

But since $x_0 = (x_t - \tfrac{b}{\theta}) e^{\theta t} + \tfrac{b}{\theta}$, we can express everything in terms of $x_t$ if needed.

---

## ✅ Summary

| Component          | Expression                                                          |
| ------------------ | ------------------------------------------------------------------- |
| **Flow**           | $x_t = (x_0 - \tfrac{b}{\theta}) e^{-\theta t} + \tfrac{b}{\theta}$ |
| **Jacobian**       | $J_t = e^{-\theta t} I$                                             |
| **Volume change**  | $\det(J_t) = e^{-d\theta t}$                                        |
| **Density change** | $p_t(x_t) = p_0(x_0) \cdot e^{d\theta t}$                           |
Assuming the initial distribution is:

$$
p_0(x_0) \sim \mathcal{N}(0, I)
$$

and the flow is governed by the vector field:

$$
u_t(x) = -\theta x + b \quad (\theta > 0, \; b \in \mathbb{R}^d)
$$

---

### 🧭 Step 1: Solve the ODE (flow map)

The solution to the ODE is:

$$
x_t = \phi_t(x_0) = (x_0 - \tfrac{b}{\theta}) e^{-\theta t} + \tfrac{b}{\theta}
$$

This is an **affine transformation** of the initial variable $x_0$, i.e.,

$$
x_t = A_t x_0 + m_t \quad \text{where}
$$

* $A_t = e^{-\theta t} I$
* $m_t = \left(1 - e^{-\theta t}\right) \tfrac{b}{\theta}$

---

### 📦 Step 2: Pushforward the Gaussian

If $x_0 \sim \mathcal{N}(0, I)$ and we apply an affine transformation:

$$
x_t = A_t x_0 + m_t
$$

then the resulting distribution is also Gaussian:

$$
x_t \sim \mathcal{N}(m_t, A_t A_t^\top) = \mathcal{N}\left(\left(1 - e^{-\theta t}\right) \tfrac{b}{\theta}, e^{-2\theta t} I \right)
$$

---

### ✅ Final Answer

$$
p_t(x) = \mathcal{N}\left(x \;\middle|\; \left(1 - e^{-\theta t}\right) \tfrac{b}{\theta},\; e^{-2\theta t} I \right)
$$

This tells us:

* The **mean** is shifting toward $\tfrac{b}{\theta}$
* The **variance** is shrinking to zero as $t \to \infty$
* In the limit, $p_t(x) \to \delta\left(x - \tfrac{b}{\theta}\right)$, as expected

Let me know if you'd like a plot showing the density evolution!

以下就是 Flow Matching.
## Flow Matching

![[Pasted image 20250509161536.png]]
![[Pasted image 20250509161714.png]]

基本是用 neural network $v_t(x;\theta)$ 近似真正的 vector field $u_t(x)$。一旦有了 $u_t(x)$ 可以得到 $p_t(x)$.  但是如何得到 $\phi_t$?  似乎是解 ODE.   但是 flow matching 做 sampling 只需要 vector field.  所以 flow 好像只是用來做數學推導？

![[Pasted image 20250509163316.png]]

### 再來就是最難理解的部分: Conditional flow matching

**先說結論：把 flow matching 變成 conditional flow matching!**

![[Pasted image 20250512001614.png]]

![[Pasted image 20250512142555.png]]

#### 推導 Conditional Flow Matching:  

**這部分和 Score matching 非常像，從 score matching 變成 transition (conditional) probability matching!**

但 flow matching 和 score matching 是不同的是：
- Diffusion noise and denoiser,  denoiser 其實就是 score function!  
- 因為 flow matching 沒有加 noise.  所以只能從 $t=1$ 的 sample $x_1$ 直接入手，反推要得到 $x_1$ 的 conditional probability 和 "conditional vector field".  重中之重就是 conditional vector field!
- 關鍵 1：$p_0(x\mid x_1) = p(x)$ at time $t=0$:  就是 $t=1$ 的 sample $x_1$ 和 $t=0$ 的 source 完全無關！
	- 這點我有點懷疑。因為 flow 的 $x_1$ 對應的 $x_0$ 是 $N(0, I)$ 沒問題。但是 independent to be verified?
- 關鍵 2：$p_1(x\mid x_1) = N(x; x_1, \sigma^2_{min} I)\approx \delta(x-x_1)$ at time $t=1$:  obvious!

#### Conditional probability: 和 diffusion 的 transition probability 相同 
這個公式有點特別： 
$$
p_t(x)=\int p_t(x \mid x_1) q(x_1) d x_1 = \int p_{t\mid 1}(x)\, q(x_1) d x_1
$$
不過我們可以驗證一下：$q(x)$  是真正的 data distribution，我們希望 $p_1(x)\approx q(x)$
 $t= 1$ 正確：
$$
p_1(x) = \int p_1(x \mid x_1)\, q(x_1)\, dx_1 \approx \int \delta(x - x_1)\, q(x_1)\, dx_1 = q(x)
$$

$t = 0$ 正確：

$$
p_0(x) = \int p_0(x \mid x_1)\, q(x_1)\, dx_1 = \int p_0(x)\, q(x_1)\, dx_1 = p_0(x)
$$

驗證 $p_t(x)$ 正確：

$$
\begin{aligned}
p_t(x) &= p(x_t) = \int p(x_t \mid x_1)\, q(x_1)\, dx_1 \\
&= \int \frac{p(x_t, x_1)}{p(x_1)}\, q(x_1)\, dx_1 \\
&= \int p(x_t, x_1)\, dx_1 = p(x_t) = p_t(x)
\end{aligned}
$$

#### Conditional Vector Field: New! 

再來最神奇的部分:  deterministic 的 vector field $u_t(x)$ 也可以有 conditional on $x_1$ vector field!  也就是從 deterministic field 變成一個 conditional distribution 的期望值！

Interestingly, we can also define a marginal vector field, by "marginalizing" over the conditional vector fields in the following sense (we assume $p_t(x)>0$ for all $t$ and $x$ ):

$$
u_t(x)=\mathbb{E}_{x_1\sim p_{1\mid t}}[u_t(x \mid x_1)] =\int u_t(x \mid x_1) \frac{p_t(x \mid x_1) q(x_1)}{p_t(x)} d x_1
$$
where
$$
\begin{aligned}
p_{1\mid t} &= p(x_1\mid x_t) = \frac{p(x_t \mid x_1) p(x_1)}{p(x_t)} \\
&= \frac{p(x_t \mid x_1) q(x_1)}{p(x_t)} \\
&= \frac{p_t(x \mid x_1) q(x_1)}{p_t(x)}
\end{aligned}
$$
為什麼要這麼繞？為什麼不是 $u_t(x)=\mathbb{E}_{x_1\sim p_{1}}[u_t(x \mid x_1)]$？還要把 $x_1 \sim p_1$ 變成 $x_1 \sim p_{1\mid t}$?
Ans:  代表我們只看從 $x_t$ 到 $x_1$ 的 trajectories, 而不是所有到 $x_1$ 到 trajectories.  是因為 causality?  如果沒有經過 $x_t$ 產生的 $x_1$ 對 vector field 毫無貢獻？

可以把 $u_t(x \mid x_1)$ 想成是「若最終目的地是 $x_1$，那麼在 $x$ 時應該往哪個方向走」。

那現在我們想要知道 $u_t(x)$ :「在不知道目的地的情況下，我應該往哪個方向走？」，這時候就應該根據所有**可能的目的地 $x_1$** 來加權平均，但這個加權要根據「給定我在 $x_t$，目的地是 $x_1$ 的機率」，也就是 $p(x_1 \mid x_t)$。

> 為什麼要這麼繞？因為 marginal vector field 的定義必須是條件期望，要考慮目前這個 $x$ 是怎麼來的。也就是「在目前的點 $x_t$ 之下，最可能的未來點是哪些 $x_1$」，而不是單純平均所有 data points。

這是 flow-based method 中極為關鍵、但常被忽略的 probabilistic insight，讓 deterministic vector field 能夠透過 probabilistic marginalization 變得更加靈活與準確。
再來最神奇的部分:  deterministic 的 vector field $u_t(x)$ 也可以有 conditional on $x_1$ vector field!  也就是從 deterministic field 變成一個 conditional distribution 的期望值！

Interestingly, we can also define a marginal vector field, by "marginalizing" over the conditional vector fields in the following sense (we assume $p_t(x)>0$ for all $t$ and $x$ ):

$$
u_t(x)=\mathbb{E}_{x_1\sim p_{1\mid t}}[u_t(x \mid x_1)] =\int u_t(x \mid x_1) \frac{p_t(x \mid x_1) q(x_1)}{p_t(x)} d x_1
$$
where
$$
\begin{aligned}
p_{1\mid t} &= p(x_1\mid x_t)  = \frac{p_t(x \mid x_1) q(x_1)}{p_t(x)}\\
\end{aligned}
$$

同樣我們看兩個 cases
$t=0$ 
$$\begin{aligned}
u_0(x)&=\int u_0(x \mid x_1) \frac{p_0(x \mid x_1) q(x_1)}{p_0(x)} d x_1\\
&=\int u_0(x \mid x_1)  q(x_1) d x_1\\
\end{aligned}$$
For OT case, $u_0(x\mid x_1)=x_1-x$
所以在 OT case, 每個 $x_0$ 都會先指向 $x_1$ 的平均值。
$$\begin{aligned}
u_0(x)&=\int u_0(x \mid x_1)  q(x_1) d x_1 = \mathbb{E}[x_1]-x\\
\end{aligned}$$
$t=1$ 
$$\begin{aligned}
u_1(x)&=\int u_1(x \mid x_1) \frac{p_1(x \mid x_1) q(x_1)}{p_1(x)} d x_1\\
&=\int u_1(x \mid x_1) \frac{\delta(x-x_1) q(x_1)}{p_1(x)} d x_1\\
&\approx u_1(x \mid x) = u_1(x)\\
\end{aligned}$$


### Conditional Flow 例子：2D Gaussian (from reference)

先定義兩個 distributions, 以下的 $\mu=10$ 
$$
\begin{gathered}
p_0=\mathcal{N}([-\mu, 0], I) \text { and } p_1=\mathcal{N}([+\mu, 0], I) \\
\text { with } \phi_t\left(x_0 \mid x_1\right)=(1-t) x_0+t x_1
\end{gathered}
$$
$$
\begin{aligned}
&\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t, q(x_1), p(x_0)}\left\|v_t(\psi_t(x_0))-\frac{d}{d t} \psi_t\left(x_0\right)\right\|^2
\end{aligned}
$$

One-sided flow matching?  Two sided flow matching?





假設 $X_1 \in$ {”貓", "狗“} 的機率為 10%, 90%.  
u(x_t)  在 t =0 是各 50%, 50%.  但是到 t = 1 是 10%, 90%,  那在 t 中間如何?

u_t = u(x_t | x_1="貓") * 

![[Pasted image 20250512001549.png]]

如何證明：重點是 conditional vector field 也滿足 continuity equation!  偏微分版。
Question: 也會滿足全微分版 continuity equation? YES, Appendix G

![[Pasted image 20250513201306.png]]

![[Pasted image 20250513200530.png]]

再來很神奇的就把 flow matching 變成 conditional flow matching.

### Why Conditional Flow Matching?

神奇的三部曲如下：
$$
\begin{aligned}
&\mathcal{L}_{\mathrm{FM}}(\theta)=\mathbb{E}_{t, p_t(x)}\left\|v_t(x)-u_t(x)\right\|^2\\
&\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t, q(x_1), p_t(x \mid x_1)}\left\|v_t(x)-u_t(x \mid x_1)\right\|^2,\\
&\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t, q(x_1), p(x_0)}\left\|v_t(\psi_t(x_0))-\frac{d}{d t} \psi_t\left(x_0\right)\right\|^2
\end{aligned}
$$
Step1:  Global flow matching
Step2:  Conditional flow match: 注意
- $\mathcal{L}_{\mathrm{FM}} \ne \mathcal{L}_{\mathrm{CFM}}$，**但是** $\min\mathcal{L}_{\mathrm{FM}} \equiv \min\mathcal{L}_{\mathrm{CFM}}$
- 所以 $u_t(x) \ne u_t\left(x \mid x_1\right)$ => global flow 和 conditional flow 可能不一致？
Step3:  轉換成可以 sample 的 distribution, $t, x_1, x_0$
- **重點是如何假設 $\psi_t(x_0)$ 和 $x_0, x_1$ 的關係。最簡單就是 linear interpolation.**


Why conditional vector field?  因為 flow matching 是 sampling from $p_t(x)$，但是 conditional flow matching 可以從 data $q(x_1)$ sampling 來 training.  
- 但是還是有 $p_t(x\mid x_1)$ 才能 training?  如同 diffusion 的 transition probability: 老把戲是 Gaussian. $p_t(x\mid x_1)\sim N(\mu_t(x_1), \sigma^2_t(x_1) I)$
- 這個 Gaussian equivalently!  $x_{t\mid 1} = \mu_t(x_1) + \sigma_t(x_1) \cdot z, \quad z\sim N(0, I)$
- 因為 $z$ 和 $x_0$ 一樣 $N(0, I)$，也可以改成：$x_{t\mid 1} = \mu_t(x_1) + \sigma_t(x_1)  x_0$
- 再因為 $x_{t\mid 1} = \phi(x_0\mid x_1)=\psi(x_0)$,  所以也可以寫：$\psi(x_0) = \mu_t(x_1) + \sigma_t(x_1)  x_0$


以下是我原來想法，缺乏 sample 細節
- 因為 $p_0(x\mid x_1) = N(0, \sigma^2_0 I)$,  $p_1(x\mid x_1) = N(x_1, \sigma^2_1 I)$. 最簡單直覺的做法是  $p_t(x\mid x_1)\approx N(t x_1, \sigma^2_t I)$.  問題是 $\sigma_0, \sigma_t, \sigma_1$ 如何設定？  $\sigma_1 = \sigma_{min}\sim0$, $\sigma_0=1$, $\sigma^2_t = (1-t)$

After reading the paper, 我發現兩個點. $\mu_t$ 不是 $u_t$
- $\mu_t(x_1) = 1-t$ 只是一種方法，稱為 optimal transport (OT)
- $\sigma_t^2(x_1)$ 似乎 depends on 其他條件 and boundary condition

Boundary condition
- $\mu_1(x_1) = x_1$, $\sigma_1(x_1)=\sigma_{min}$
- $\mu_0(x_1) = 0$, $\sigma_0(x_1)=1$

#### Sample 的細節：把 condition vector field 轉換成 condition flow 的微分

定義一個 conditional flow (**注意不是 vector field!**) on $x_1=X_1$.   這是一個線性，但 gain and offset 隨時間變化。

(Wrong!) 可以和之前簡單的 time-independent 線性 (affine vector field) vector field 比較。 之前的 affine 是 vector field!  Flow 是一堆的軌跡，vector field 是斜率。兩者完全不同！

![[Pasted image 20250512165713.png]]
上式的 $x$ 基本是 $x_0$.  其他所有的 flow, mean, variance 都是 conditional on $x_1 =X_1$.
比較好的 conditional flow 寫法是:

$$
\psi_t(x) = \psi_t(x_0) = x_t = \sigma_t(x_1) x_0 + \mu_t(x_1)
$$

對應的 conditional vector field 是：
$$u_t(x\mid x_1) = \frac{\psi_t(x_0)}{dt} = \frac{d x_t}{dt} = \dot{\sigma}_t(x_1) x_0 + \dot{\mu_t}(x_1)$$
同樣比較好的 conditional vector field 寫法是:

$$u_t(x\mid x_1) = u(x_t\mid x_1) = \frac{\psi_t(x_0)}{dt} = \frac{d x_t}{dt} = \dot{\sigma}_t(x_1) x_0 + \dot{\mu_t}(x_1)$$
一般用 $x_0$ 比較好，因爲可以用來 sampling from $\mathcal{N}(0, I)$ 做 flow matching training!

但也可以把 $x_0$  換成 $x_t$ 用上面的 Gaussian，如此得到 instant conditional vector field at $t$.
$$\begin{aligned}
u_t(x\mid x_1) &= u(x_t\mid x_1) = \frac{\psi_t(x_0)}{dt} = \frac{d x_t}{dt} = \dot{\sigma}_t(x_1) x_0 + \dot{\mu_t}(x_1)\\
&= \dot{\sigma}_t(x_1) \left[\frac{x_t - u_t(x_1)}{\sigma_t(x_1)}\right] + \dot{\mu_t}(x_1) \\
&= \frac{\dot{\sigma}_t(x_1)}{\sigma_t(x_1)} (x - \mu_t(x_1))+ \dot{\mu_t}(x_1) \\
\end{aligned}$$

**上式的好處是如果我們已經知道 $p_t$ 在時間和空間的分佈, i.e. $p(x, t)$ from Fokker-Planck equation，可以直接轉換成 flow!!**
$$p_t(x\mid x_1)\sim N(\mu_t(x_1), \sigma^2_t(x_1) I)$$


這個 conditional flow 用圖看比較清楚。
最後 reach $x_1=X_1$, 從一個 fat initial condition ($\sigma_t(x_1)$ 隨時間變小)，但是最終收斂到 $\mu_t(x_1)=X_1$
![[Pasted image 20250514121948.png]]

$\psi_0(x) = \psi_0(x_0) = \sigma_0(x_1) x_0 + \mu_0(x_1) = x_0 = x$ 如 flow 需要 $t=0$ 座標不變
$\psi_1(x) = \psi_1(x_0) = \sigma_1(x_1) x_0 + \mu_1(x_1) = \sigma_{min} x_0 + X_1 \sim X_1$  基本所有的 $x_0$ 都被壓縮到 $X_1$, 因爲 conditional on $x_1=X_1$

有了 conditional flow, 可以微分得到 conditional vector field!  這也是最後要做的 flow matching!

![[Pasted image 20250512175106.png]]
#### 如何 sample：$\psi(x_0) = \mu_t(x_1) + \sigma_t(x_1)  x_0$ and $\frac{d}{dt}\psi_t(x_0)$?

$t \sim [0, 1]$.
$x_1$ 是直接從 data set sample 的 image $q(x_1)$.
$x_0 \sim N(0, I)$ 也非常簡單。

#### 其實最後關鍵就是如何選兩個參數
$\mu_t(x_1), \sigma_t(x_1)$ and $\dot{\mu}_t(x_1), \dot{\sigma}_t(x_1)$  
with boundary condition
- $\mu_1(x_1) = x_1$, $\sigma_1(x_1)=\sigma_{min}$
- $\mu_0(x_1) = 0$, $\sigma_0(x_1)=1$

我們看一些例子。

#### Example I: Optimal Transport (OT) conditional VF (Vector Field)

最簡單就是線性內差：
$\mu_t(x_1) = t x_1$, $\sigma_t(x_1) = 1-(1-\sigma_{min})\,t$
$\psi_t(x_0) = x_t = t x_1 + (1-(1-\sigma_{min})t) x_0$

**對應的 condition vector field，物理意義非常簡單，就是一個 constant field 和 sampled 的 $t$ 無關！而且是 $x_0$ 和目標的$x_1$ 的向量差！就是一路直衝終點！**
$$\frac{\psi_t(x_0)}{dt} = \frac{d x_t}{dt} = x_1 - (1-\sigma_{min})x_0\approx x_1 - x_0$$

如果以 $x_t$ local or instant 角度的 conditional vector field:
$$\begin{aligned} u_t(x\mid x_1) = u(x_t\mid x_1) &= x_1 + (1-\sigma_{min}) \frac{x_t - t x_1}{1-(1-\sigma_{min})t}\\ 
&=  \frac{x_1 - (1-\sigma_{min}) x_t}{1-(1-\sigma_{min})t}\\
&=  \frac{x_1 - (1-\sigma_{min}) x}{1-(1-\sigma_{min})t}\\
\end{aligned}$$

**OT Summary**
$t = 1$
BC (boundary condition):  $\mu_1(x_1) = x_1$, $\sigma_1(x_1)=\sigma_{min}$
conditional flow: $\psi_1(x_0) = x_1 = x_1 + \sigma_{min} x_0 \approx x_1$, mean and variance aligned with BC
conditional vector field: $u_1(x \mid x_1)=u(x_1\mid x_1)=\frac{(\sigma_{min} x_1)}{\sigma_{min}} =x_1$,  好像有點怪怪的

假設 $t=1-\Delta t$
$$\begin{aligned} u_{1-\Delta t}(x\mid x_1) &= u(x_{1-\Delta t}\mid x_1) =  \frac{x_1 - (1-\sigma_{min}) x_{1-\Delta t}}{1-(1-\sigma_{min})(1-\Delta  t)}\\
&\approx\frac{x_1 - x_{1-\Delta t} + \sigma_{min} x_{1-\Delta t}}{\Delta  t +\sigma_{min}}\\
\end{aligned}$$
所以在 $\Delta t$ 比較大的時候，$u_{1-\Delta t} \approx \frac{d x_1}{d t}$, 還是 flow 在 dominate.  
但等到 $\Delta t$ 接近無窮小，$u_{1-\Delta t} \approx x_1$, 就是指到 $x_1$


$t = 0$
BC (boundary condition):  $\mu_0(x_1) = 0$, $\sigma_0(x_1)=1$
conditional flow: $\psi_0(x_0) = x_0 \sim N(0, I)$, aligned with boundary condition
conditional vector field: $u_0(x \mid x_1)=u(x_0\mid x_1)=x_1-(1-\sigma_{min})x_0\approx x_1 - x_0$

這個部分的結果和之前抵觸！！$u_0(x \mid x_1)=x_0$  因爲 $x_0$  和 $x_1$  完全不相關！但是在 OT 的情況變成完全相關！



OT 的物理意義是 $v_t(x_t)$ (neural network vector field) 在任何時間的 vector field 就是 $x_1-x_0$，assuming $\sigma_{min} \approx 0$. 非常簡單到不像話！！


![[Pasted image 20250514145237.png]]

### Diffusion: 從 Transition Probability Distribution 得到 Conditional Vector Field!!!!

Diffusion model 都會有 transition/condition probability distribution
$$p_t(x\mid x_0)\sim N(\mu_t(x_0), \sigma^2_t(x_0) I)$$
Diffusion 的 $x_0$ 是無 noise (or 很小 noise) 的 image,  相當於 flow matching 的 $x_1$.   另外 diffusion 的時間是 $T$.  沒有一個固定數字，所以需要轉換一下。

如果可以轉換出  condtion probability:
$$p_t(x\mid x_1)\sim N(\mu_t(x_1), \sigma^2_t(x_1) I)$$
Conditional vector field:
$$\begin{aligned}
u_t(x\mid x_1) = \frac{\dot{\sigma}_t(x_1)}{\sigma_t(x_1)} (x - \mu_t(x_1))+ \dot{\mu_t}(x_1) \\
\end{aligned}$$
**檢查上式 $p_t, u_t$ 滿足 condition continuity equation**

1. Partial derivative conditional continuity equation: see Appendix F
$$\begin{aligned}-\frac{\partial  p_t(x\mid x_1)}{\partial t} &= \nabla \cdot [u_t(x\mid x_1) p_t(x\mid x_1)]\\
&= p_t \left[ \text{d} \frac{\dot{\sigma}_t}{\sigma_t} - \frac{(x - \mu_t) \cdot \dot{\mu}_t}{\sigma_t^2} - \frac{\|x - \mu_t\|^2 \dot{\sigma}_t}{\sigma_t^3} \right]
\end{aligned}$$

2. Total derivative conditional continuity equation: see Appendix G
$$\begin{aligned}-\frac{d \log p_t(x\mid x_1)}{dt} = \nabla \cdot u_t(x\mid x_1) =\text{d}\,\frac{\dot{\sigma}_t(x_1)}{\sigma_t(x_1)} \quad \text{where d is the dimension } \end{aligned}$$

**注意偏微分和全微分的差別**
$$\frac{d f(\mathbf{x},t)}{dt} = \frac{\partial f(\mathbf{x},t)}{\partial t} + \frac{d\mathbf{x}}{dt}\cdot \nabla f(\mathbf{x},t) = (\frac{\partial}{\partial t} + \frac{d\mathbf{x}}{dt}\cdot \nabla) f(\mathbf{x},t) =  (\frac{\partial}{\partial t} + \mathbf{u}\cdot \nabla) f(\mathbf{x},t)$$
**Continuity equation**
1. **Eulerian form (偏微分)**: $\partial_t p +\nabla\cdot(p\,u)=0.$
2. **Lagrangian form (全微分)**: $\displaystyle -\frac{d\log p}{dt}=-\bigl(\partial_t+u\cdot\nabla\bigr)\log p= \nabla\cdot u,$

Lagrangian form 很有意思把 Flow and Score function 連結一起：
$-\partial_t \log p = u\cdot\nabla\log p + \nabla\cdot u,$



#### Example II: VE (Variance Explode) Diffusion conditional VF

如果我們有 VE 的 probability form, 就可以得到 vector field

![[Pasted image 20250518201728.png]]

$\mu_t(x_1), \sigma_t(x_1)$ and $\dot{\mu}_t(x_1), \dot{\sigma}_t(x_1)$  
with boundary condition
- $\mu_1(x_1) = x_1$, $\sigma_1(x_1)=\sigma_{min}$
- $\mu_0(x_1) = x_1$, $\sigma_0(x_1)=\sigma_{max}$
$$
\psi_t(x) = \psi_t(x_0) = x_t = \sigma_t(x_1) x_0 + \mu_t(x_1)
$$


#### Example III: VP (Variance Preserve) Diffusion conditional VF

![[Pasted image 20250518204128.png]]
以上推導見 Appendix E.



因為要把 flow 變成 diffusion, 所以有點複雜，我們先看結果：
![[Pasted image 20250514151000.png]]
![[Pasted image 20250514151045.png]]

關鍵是 $\alpha_t$ 的定義如下： $\beta(t)$ 是 scaling function,  是遞增函數。  $\beta(0)=0$?
![[Pasted image 20250514151153.png]]

$\alpha_t$ 是遞減函數，$\alpha(0)=1$, $\alpha(1)\to 0$。
$\alpha_{1-t}$ 是遞增函數，$\alpha_{1-t}(0)\to 0$, $\alpha_{1-t}(1)\to 1$。


### 檢討

**有點像粒子和坡。  OT 是粒子，Diffusion 比較像坡！！**



> [!NOTE]
> 對比 VE SDE/ODE 的形式：
> $\sigma_t$ 是加在原始 image $\mathbf{x}_0$:    $\mathbf{x}_t = \mathbf{x}_0 + \sigma_t \,\mathbf{z}_t$.  Diffusion 定義的 t=0 是 flow 定義 t=1. 
> 其中 $\sigma_t\ge 0$ 而且是**遞增函數！稱為 noise scheduling function.**  我們可以假設 $\sigma(0)=0$ 對應原始無 noise images $\mathbf{x}_0$.
> 
> VE ODE 看起來非常簡潔： $d \mathbf{x}_t =  - \frac{1}{2} g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})dt = - \sigma(t) \dot{\sigma(t)}\nabla_{\mathbf{x}} \log p_t(\mathbf{x})dt \approx - \sigma(t) \dot{\sigma(t)}\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)dt$
> 


所以 $\mu_t(x_1) = x_1$ ($x_0$ in diffusion), 也就是 $\dot{u}_t(x_t) = 0$   ($f=0$ in diffusion) 也就是 image 的强度不變。但是 noise 從非常大到非常小。

flow match $\sigma_1 = \sigma_{min}$  對應 diffusion $\sigma_0 = \sigma_{min}$
flow match $\sigma_0 = \sigma_{max}$  對應 diffusion $\sigma_T = \sigma_{max} \gg x_1$



我們可以對比 VE SDE 的 SDE and ODE 形式：


**Forward SDE: for training.**
$$
d \mathbf{x}_t={\boldsymbol{f}}(\mathbf{x}_t, t)\, d t+g(t)\, d \mathbf{w}_t,\quad \text{ with } d \mathbf{w}_t \sim N(0, d t)
$$

**Reverse SDE: for sampling.**  下式的 $dt$ 是**負**無窮小 time step.
$$
d \mathbf{x}_t=[{\boldsymbol{f}}(\mathbf{x}_t, t)-g^2(t)\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)]\, d t+g(t)\, d \mathbf{w}_t,\quad \text{ with } d \mathbf{w}_t \sim N(0, d t)
$$

**Equivalent Fokker-Planck ODE**
$$
d \mathbf{x}_t=[{\boldsymbol{f}}(\mathbf{x}_t, t)-\frac{1}{2} g^2(t)\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)]\, d t
$$
VE SDE:  $f=0, g(t)= \sqrt{\frac{d\sigma^2(t)}{dt}}$
接下來把 discrete recursive 形式變成無窮小 time step 得到 continuous SDE forward path: 

$$
\mathrm{d}\mathbf{x}_t = \sqrt{ \frac{ \mathrm{d}\left[\sigma^2_t\right]}{\mathrm{d}t} } \, \mathrm{d}\mathbf{w}_t
$$
注意這裏的 $\sigma_t$ 是加在原始 image $\mathbf{x}_0$:    $\mathbf{x}_t = \mathbf{x}_0 + \sigma_t \,\mathbf{z}_t$.
其中 $\sigma_t\ge 0$ 而且是**遞增函數！稱為 noise scheduling function.**  我們可以假設 $\sigma(0)=0$ 對應原始無 noise images $\mathbf{x}_0$.

- $f(\mathbf{x}_t, t) = 0$； $g(t) = \sqrt{ \frac{ \mathrm{d}\left[\sigma^2(t)\right]}{\mathrm{d}t} }$；or $g^2(t) = 2 \sigma(t) \dot{\sigma}(t)$ 

VE reverse SDE (sampling): $d \mathbf{x}_t = - g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})dt+ g(t) \,d\mathbf{w}_t = - 2 \sigma(t) \dot{\sigma(t)}\nabla_{\mathbf{x}} \log p_t(\mathbf{x})dt+ \sqrt{2 \sigma(t) \dot{\sigma(t)}} \,d\mathbf{w}_t$

VE ODE 看起來非常簡潔： $d \mathbf{x}_t =  - \frac{1}{2} g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})dt = - \sigma(t) \dot{\sigma(t)}\nabla_{\mathbf{x}} \log p_t(\mathbf{x})dt \approx - \sigma(t) \dot{\sigma(t)}\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)dt$




## Reference

MIT 6.S184: Flow Matching and Diffusion Models https://www.youtube.com/watch?v=GCoP2w-Cqtg&t=28s&ab_channel=PeterHolderrieth

Yaron Meta paper: [[2210.02747] Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

An Introduction to Flow Matching: https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html

## Appendix A:  


首先，原方程是一个对流-扩散方程：

$$
\frac{\partial p(x,t)}{\partial t} = -\nabla \cdot [\mathbf{u}(x,t) p(x,t)] + D(t) \Delta p(x,t)
$$

用户给出的 (Wrong!) 第二个方程是：

$$
\frac{d \log p(x,t)}{dt} = -\nabla \cdot \mathbf{u}(x,t) + \nabla \cdot [D(t) \nabla \log p(x,t)]
$$

我们需要验证第二个方程是否正确。

将 $p(x,t)$ 表示为 $\exp(\phi(x,t))$，即 $\phi(x,t) = \log p(x,t)$，则 $p = e^\phi$。代入原方程：

1. 左边项：
   $$
   \frac{\partial p}{\partial t} = e^\phi \frac{\partial \phi}{\partial t}
   $$

2. 右边对流项：
   $$
   -\nabla \cdot [\mathbf{u} p] = -\nabla \cdot [\mathbf{u} e^\phi] = -e^\phi (\nabla \cdot \mathbf{u} + \mathbf{u} \cdot \nabla \phi)
   $$

3. 右边扩散项：
   $$
   D(t) \Delta p = D(t) \nabla \cdot (e^\phi \nabla \phi) = D(t) e^\phi (|\nabla \phi|^2 + \Delta \phi)
   $$

将原方程两边除以 $e^\phi$ 得到：
$$
\frac{\partial \phi}{\partial t} = -\nabla \cdot \mathbf{u} - \mathbf{u} \cdot \nabla \phi + D(t) (|\nabla \phi|^2 + \Delta \phi)
$$

考虑物质导数（全导数）：
$$
\frac{d \phi}{dt} = \frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla \phi
$$

代入上式：
$$
\frac{d \phi}{dt} = -\nabla \cdot \mathbf{u} + D(t) (|\nabla \phi|^2 + \Delta \phi)
$$

用户给出的方程是：
$$
\frac{d \log p(x,t)}{dt} = -\nabla \cdot \mathbf{u} + \nabla \cdot [D(t) \nabla \log p(x,t)]
$$

其中右边扩散项为：
$$
\nabla \cdot [D(t) \nabla \phi] = D(t) \Delta \phi
$$

比较两者的结果，正确的结果中包含 $D(t) (|\nabla \phi|^2 + \Delta \phi)$，而用户的结果中缺少了 $D(t) |\nabla \phi|^2$ 项，因此用户的方程不正确。

### 最终答案



Excellent question — and you're now honing in on a very clean and insightful formulation. Let's unpack it carefully.

You're proposing:

$$
\frac{d}{dt} \log p(x(t), t)
= -\nabla \cdot f + \frac{1}{2} g(t)^2 \nabla \cdot \nabla \log p = -\nabla \cdot f + \frac{1}{2} g(t)^2 \nabla^2 \log p
$$

This is **almost correct**, and it actually *is* correct under a specific assumption: when you treat the total derivative deterministically (ignoring Itô correction terms that come from stochastic calculus). Let's see why, and when it's valid or not.

---

## 🧮 Two Ways to Think About the Total Derivative

### ✅ 1. **Expected (mean-field) evolution** — what the **density** evolves like

If you're **tracking how $\log p(x(t), t)$ evolves on average**, then using the **Fokker–Planck equation** and applying the chain rule gives:

$$
\frac{d}{dt} \log p(x(t), t)
= \partial_t \log p + f \cdot \nabla \log p + \frac{1}{2} g(t)^2 \left( \nabla^2 \log p + \|\nabla \log p\|^2 \right)
$$

> This is the **Itô formula** applied to $\log p$.
> The term $\|\nabla \log p\|^2$ arises due to the nonlinear transformation (log of a stochastic function).

---

### 🚫 2. What you're proposing:

$$
\frac{d}{dt} \log p = -\nabla \cdot f + \frac{1}{2} g(t)^2 \nabla^2 \log p
$$

This is actually the **Eulerian time derivative** of $\log p$, derived directly from the **Fokker–Planck equation** (dividing both sides by $p$).
But **this is not the same** as the **Itô total derivative along a trajectory**.

Let me show it explicitly:

---

## 📘 Derivation from Fokker–Planck:

Start from:

$$
\partial_t p = -\nabla \cdot (f p) + \frac{1}{2} g(t)^2 \nabla^2 p
$$

Divide both sides by $p$:

$$
\partial_t \log p = -\nabla \cdot f - f \cdot \nabla \log p + \frac{1}{2} g(t)^2 \left( \nabla^2 \log p + \|\nabla \log p\|^2 \right)
$$

Now take the **total derivative**:

$$
\frac{d}{dt} \log p = \partial_t \log p + \dot{x} \cdot \nabla \log p
$$

Plug in $\dot{x} = f + \text{noise}$. The expected value (ignoring stochastic term) is:

$$
f \cdot \nabla \log p
$$

So when you add $f \cdot \nabla \log p$ to the expression above, the two $f \cdot \nabla \log p$ terms cancel:

$$
\boxed{
\frac{d}{dt} \log p = -\nabla \cdot f + \frac{1}{2} g(t)^2 \left( \nabla^2 \log p + \|\nabla \log p\|^2 \right)
}
$$

So the correct form **must** include the $\|\nabla \log p\|^2$ term due to the **Itô correction**.

---

## ✅ Conclusion

So to answer directly:

> **Why not**

$$
\frac{d}{dt} \log p = -\nabla \cdot f + \frac{1}{2} g^2 \nabla^2 \log p?
$$

Because it **misses the nonlinear Itô term** $\frac{1}{2} g^2 \|\nabla \log p\|^2$, which is **required** when taking the total derivative of a nonlinear function of a stochastic process (like $\log p$).

This extra term comes from:

$$
\frac{1}{2} \text{Tr}(g^2 H_{\log p}) + \frac{1}{2} \| g \nabla \log p \|^2
$$

If you're okay with expectation-level or weak approximations, your version might suffice. But for **pathwise correctness**, you need the full Itô expression.

---

Would you like this rewritten in terms of the **score function** $s(x, t) = \nabla \log p(x, t)$? It often makes these formulas neater.



### Conclusion

Both pure diffusion and Ornstein-Uhlenbeck processes with Gaussian solutions satisfy their respective Fokker–Planck equations and the log-probability total derivative (Lagrangian) formulation. This provides a strong analytical foundation for diffusion modeling, score-based generative models, and reverse-time SDE analysis.


