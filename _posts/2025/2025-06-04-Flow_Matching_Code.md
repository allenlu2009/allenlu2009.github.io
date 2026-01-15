---
title: Math AI - Expand Score Matching to Flow Matching
date: 2025-05-05 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2025-0505-Flow_Matching]]"
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






### Flow $\phi_t$ = 連續的 (自己到自己) 座標變換
**Flow** $\phi_t$ 定義:  $x_t \triangleq \phi_t\left(x_0\right)$,  顯然 $t=0 \to x_0 = \phi_0\left(x_0\right)$    $\phi_t(x)$ 必須可微分且可逆。

(重要) 同時 $x_t$  也要滿足運動方程：$\frac{d x_t}{d t}=u_t(x_t)$
所以會得到以下的方程式：
$$\frac{d \phi_t(x_0)}{d t}=u_t\left(\phi_t\left(x_0\right)\right)$$
因為 $\phi_t(x)$ 是自己到自己的座標變換，在某些情況，可以省掉 subscript.

$$\frac{d \phi_t(x)}{d t}=u_t\left(\phi_t\left(x\right)\right)$$
有無窮多的 $\phi_t$ and $u_t$ 滿足 $p_0(x)$ 到 $p_1(x)$ 的分佈轉換。最直接而且簡單的就是 linear interpolation flow.  
- Linear flow:  從 $x_0$ 的角度，是從 $x_0$ 到 $x_1$ 是直線，Wrong!  因為這代表從 $x_0$ 開始，遇到一個固定, constant 的 vector field ($u_t(x)$).  這代表 output distribution 和 input distribution 一樣，最多是 mean shift! 這種固定 and deterministic vector field 沒有太大的用處。
	- 即使同樣的 distribution with shift mean, 我們可以看到 $u_t(x)$ 也不是 constant vector.
- 退而求其次，是從 $x_1$ 的角度， given $x_1$，對應的 $x_0$ 是直線。另外不是一個 $x_0$ 而是一個 distribution.  其 conditional vector field 是固定, constant 值。 

以上太抽象，我們看實際的例子。


### Training ($t, x_0, x_1 \to u_t(x_t)$)
Flow matching 神奇三部曲如下：
**Method 1 (直接法):  Global flow matching**
$$
\begin{aligned}
&\mathcal{L}_{\mathrm{FM}}(\theta)=\mathbb{E}_{t, p_t(x)}\left\|v_t(x)-u_t(x)\right\|^2\\
\end{aligned}
$$
- 一般 $p_0 \sim N(0, I)$, 但是 $p_1$ 未知，所以 $p_t$  也未知。
- 除非是非常簡單的 $p_1$，同時用 linear interpolation $x_t = t x_0 + (1-t) x_1$ , 可以直接計算 $u_t(x)$.  Really, how?  我覺得還是要用 condition flow 的定義！

**Method 2 (間接法):  Conditional flow match:** 
$$
\begin{aligned}
&\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t, q(x_1), p_t(x \mid x_1)}\left\|v_t(x)-u_t(x \mid x_1)\right\|^2,\\
\end{aligned}
$$
- 此時需要 $p_t(x\mid x_1)$，一般假設 Gaussian.  同時利用  linear interpolation $x_t \mid x_1 = t x_0 + (1-t) x_1$ ,  應該可以導出這個 Gaussian 的 close-form，或是可以用來計算 conditional flow.  但是無法得到 marginal flow 因爲 $p_1$ 是未知。

注意
- $\mathcal{L}_{\mathrm{FM}} \ne \mathcal{L}_{\mathrm{CFM}}$，**但是** $\min\mathcal{L}_{\mathrm{FM}} \equiv \min\mathcal{L}_{\mathrm{CFM}}$
- 所以 $u_t(x) \ne u_t\left(x \mid x_1\right)$ => global flow 和 conditional flow 可能不一致？


**Method 3 (間接間接法):  轉換成可以 sample 的 distribution, $t, x_1, x_0$**
- **重點是如何假設 $\psi_t(x_0)$ 和 $x_0, x_1$ 的關係。最簡單就是 linear interpolation.**
$$
\begin{aligned}
&\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t, q(x_1), p(x_0)}\left\|v_t(\psi_t(x_0))-\frac{d}{d t} \psi_t\left(x_0\right)\right\|^2
\end{aligned}
$$
- 這個表示把 $p_0$ and $x_0$ 帶出來。不過和 method 2 應該等價。

### Inferencing/Sampling ($x_0, u_t, t \to x_1$)

此時和 conditional flow $u_t(x\mid x_1)$ 完全無關，因爲我們沒有 $x_1.$

一旦有 $u_t(x)$ 或是其近似 $v_t(x)$，就可以 sample. 
- 先從 $p_0$ randomly sample.
- 利用 $u_t(x)$ 可以逐步得到 $p_1$ 的 sample.


### Compute Likelihood  ($x_1, u_t, t \to x_0$)

應該是 sample 的反向。所以只要把 $u_t$ 反向就可以?





#### Conditional Vector Field: New! 

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


## 利用 correlation matrix



### **Step 1: Define Joint Distribution**
The vector $\begin{bmatrix} \mathbf{x}_0 \\ \mathbf{x}_t \end{bmatrix}$ is jointly Gaussian since $\mathbf{x}_t$ is a linear combination of $\mathbf{x}_0$ and $\mathbf{x}_1$. Compute its moments:  
- **Means**:  
  $$
  \mathbb{E}[\mathbf{x}_0] = -\boldsymbol{\mu}, \quad \mathbb{E}[\mathbf{x}_t] = (1-t)(-\boldsymbol{\mu}) + t(\boldsymbol{\mu}) = (2t-1)\boldsymbol{\mu}.
  $$  
- **Covariances**:  
  $$
  \text{Cov}(\mathbf{x}_0) = \mathbf{I}, \quad \text{Cov}(\mathbf{x}_t) = (1-t)^2\mathbf{I} + t^2\mathbf{I} = \sigma_t^2 \mathbf{I}, \quad \sigma_t^2 = 2t^2 - 2t + 1.
  $$  
- **Cross-Covariance**:  
  $$
  \text{Cov}(\mathbf{x}_0, \mathbf{x}_t) = \mathbb{E}[(\mathbf{x}_0 + \boldsymbol{\mu})(\mathbf{x}_t - (2t-1)\boldsymbol{\mu})^\top] = (1-t)\mathbf{I},
  $$  
  since $\mathbf{x}_1 - \boldsymbol{\mu}$ is independent of $\mathbf{x}_0 + \boldsymbol{\mu}$ and has zero mean.

### **Step 2: Apply Gaussian Conditioning Formula**
For jointly Gaussian vectors $\begin{bmatrix} \mathbf{a} \\ \mathbf{b} \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \boldsymbol{\mu}_a \\ \boldsymbol{\mu}_b \end{bmatrix}, \begin{bmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end{bmatrix} \right)$,  
$$
\mathbb{E}[\mathbf{a} \mid \mathbf{b} = \mathbf{x}] = \boldsymbol{\mu}_a + \Sigma_{ab} \Sigma_{bb}^{-1} (\mathbf{x} - \boldsymbol{\mu}_b).
$$  
Here, $\mathbf{a} = \mathbf{x}_0$, $\mathbf{b} = \mathbf{x}_t$, and:  
$$
\boldsymbol{\mu}_a = -\boldsymbol{\mu}, \quad \boldsymbol{\mu}_b = (2t-1)\boldsymbol{\mu}, \quad \Sigma_{ab} = (1-t)\mathbf{I}, \quad \Sigma_{bb} = \sigma_t^2 \mathbf{I}.
$$

### **Step 3: Substitute and Simplify**
$$
\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}] = -\boldsymbol{\mu} + \left[(1-t)\mathbf{I}\right] \left[\sigma_t^2 \mathbf{I}\right]^{-1} (\mathbf{x} - (2t-1)\boldsymbol{\mu}).
$$  
Since $\left[\sigma_t^2 \mathbf{I}\right]^{-1} = \frac{1}{\sigma_t^2} \mathbf{I}$:  
$$
= -\boldsymbol{\mu} + \frac{1-t}{\sigma_t^2} (\mathbf{x} - (2t-1)\boldsymbol{\mu}).
$$

### **Final Result**
$$
\boxed{\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}] = -\boldsymbol{\mu} + \dfrac{1-t}{\sigma_t^{2}} \left( \mathbf{x} - (2t-1)\boldsymbol{\mu} \right)}
$$

### **Intuition**  
The term $\frac{1-t}{\sigma_t^2}$ represents the **regression coefficient** adjusting for the correlation between $\mathbf{x}_0$ and $\mathbf{x}_t$. The expression linearly combines the prior mean $-\boldsymbol{\mu}$ with the deviation of $\mathbf{x}$ from the marginal mean $(2t-1)\boldsymbol{\mu}$, scaled by the relative variance contribution of $\mathbf{x}_0$ to $\mathbf{x}_t$.


## 例一：Gaussian-to-Gaussian (from Cambridge blog)

先定義兩個 2D Gaussian distributions, 以下的 $\mu=10$ 
$$
\begin{gathered}
p_0=\mathcal{N}([-\mu, 0], I) \text { and } p_1=\mathcal{N}([+\mu, 0], I) \\
\end{gathered}
$$
Given:
- **Prior distribution $p_0$:** $\mathcal{N}(\mathbf{x}; -\boldsymbol{\mu}, \mathbf{I})$,
- **Target distribution $p_1$:** $\mathcal{N}(\mathbf{x}; +\boldsymbol{\mu}, \mathbf{I})$,
OT (Optimal Transport, 就是線性內差)
- **Linear interpolation path:** $\mathbf{x}_t = t \mathbf{x}_1 + (1-t) \mathbf{x}_0$.
- Interpolation distribution $p_t$: $\mathcal{N}(\mathbf{x}; (2t-1)\boldsymbol{\mu}, \sigma^2_t\,\mathbf{I})$,  mean 是內差，variance 是 scaled 平方和 
- Mean of $\mathbf{x}_t$:
$$
\mathbb{E}[\mathbf{x}_t] = (1-t)\mathbb{E}[\mathbf{x}_0] + t\mathbb{E}[\mathbf{x}_1] = (1-t)(-\boldsymbol{\mu}) + t(\boldsymbol{\mu}) = (2t - 1)\boldsymbol{\mu}
$$
- Covariance of $\mathbf{x}_t$:
$$
\text{Cov}(\mathbf{x}_t) = (1-t)^2 \text{Cov}(\mathbf{x}_0) + t^2 \text{Cov}(\mathbf{x}_1) = (1-t)^2 \mathbf{I} + t^2 \mathbf{I} = \sigma_t^2 \mathbf{I}
$$
				where $\sigma_t^2 = (1-t)^2 + t^2 = 2t^2 - 2t + 1$.


We want to compute the **vector field $\mathbf{u}_t(\mathbf{x})$** that transports samples from $p_0$ to $p_1$.


### 找 $u_t(x)$ 對應 real-life 的 training

Method 1 利用 global flow
$$
\begin{aligned}
&\mathcal{L}_{\mathrm{FM}}(\theta)=\mathbb{E}_{t, p_t(x)}\left\|v_t(x)-u_t(x)\right\|^2\\
\end{aligned}
$$
- 假設 OT:  $x_t =(1-t) x_0+t x_1$,  因為 $x_0, x_1$ 都是 Gaussians, 可以直接計算 $p_t(x)$

直接計算 $u_t$


另一個是 conditional flow

$$
\begin{aligned}
&\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t, q(x_1), p(x_0)}\left\|v_t(\psi_t(x_0))-\frac{d}{d t} \psi_t\left(x_0\right)\right\|^2
\end{aligned}
$$
---

## Method 1:  直接計算 Vector Field, 不用 Condition VF. 

其實還是用 condition flow 加上 OT,  只是直接計算 $u_t(x)$!

### Key Insight:
The linear interpolation path is defined for pairs $(\mathbf{x}_0, \mathbf{x}_1)$, where $\mathbf{x}_0 \sim p_0$ and $\mathbf{x}_1 \sim p_1$. Lipman's method uses the **independent coupling**, where $\mathbf{x}_0$ and $\mathbf{x}_1$ are sampled independently. The conditional vector field for a given pair is:
$$
\mathbf{v}_t(\mathbf{x}_t \mid \mathbf{x}_0, \mathbf{x}_1) = \frac{d\mathbf{x}_t}{dt} = \mathbf{x}_1 - \mathbf{x}_0.
$$
The marginal vector field $\mathbf{u}_t(\mathbf{x})$ is the conditional expectation:
$$
\mathbf{u}_t(\mathbf{x}) = \mathbb{E}_{p(\mathbf{x}_0, \mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x})} \left[ \mathbf{x}_1 - \mathbf{x}_0 \right].
$$

### Derivation:
1. **Marginal Distribution at Time $t$**: (很直觀)
   The interpolation $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$ with independent $\mathbf{x}_0 \sim \mathcal{N}(-\boldsymbol{\mu}, \mathbf{I})$ and $\mathbf{x}_1 \sim \mathcal{N}(+\boldsymbol{\mu}, \mathbf{I})$ gives:
   $$
   \mathbf{x}_t \sim \mathcal{N}\left(\boldsymbol{\mu}_t, \sigma_t^2 \mathbf{I} \right),
   $$
   where $\boldsymbol{\mu}_t=(2t-1)\boldsymbol{\mu}$ and  $\sigma_t^2 = (1-t)^2 + t^2 = 2t^2 - 2t + 1$.

2. **Conditional Expectations** (Appendix D, 最關鍵的一步):
   Using Gaussian conditioning, the posterior expectations given $\mathbf{x}_t = \mathbf{x}$ are:
   $$
   \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}] = -\boldsymbol{\mu} + \frac{1-t}{\sigma_t^2} (\mathbf{x} - (2t-1)\boldsymbol{\mu})=-\boldsymbol{\mu} + \frac{1-t}{\sigma_t^2} (\mathbf{x} - \boldsymbol{\mu}_t),
   $$
Check
	- $t=1$,   $\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}_1] = \mathbb{E}[\mathbf{x}_0] = -\boldsymbol{\mu}$   (因為 independent, variance to I)
	- $t=0$,   $\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}_0] = \mathbb{E}[\mathbf{x}_0\mid \mathbf{x}_0] = \mathbf{x}_0$   (因為 deterministic, $\mu_t=-\mu$, $\sigma^2_t=1$)

$$
   \mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}] = +\boldsymbol{\mu} + \frac{t}{\sigma_t^2} (\mathbf{x} - (2t-1)\boldsymbol{\mu}).
$$
4. **Vector Field**:
   $$
   \mathbf{u}_t(\mathbf{x}) = \mathbb{E}[\mathbf{x}_1 - \mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}] = \mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}] - \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}].
   $$
   Substituting the expressions:
   $$
   \mathbf{u}_t(\mathbf{x}) = \left[ +\boldsymbol{\mu} + \frac{t}{\sigma_t^2} (\mathbf{x} - (2t-1)\boldsymbol{\mu}) \right] - \left[ -\boldsymbol{\mu} + \frac{1-t}{\sigma_t^2} (\mathbf{x} - (2t-1)\boldsymbol{\mu}) \right].
   $$
   Simplifying:
   $$
   \mathbf{u}_t(\mathbf{x}) = 2\boldsymbol{\mu} + \frac{2t-1}{\sigma_t^2} (\mathbf{x} - (2t-1)\boldsymbol{\mu}).
   $$
   如果非對稱 means, 應該和 mean difference 差有關。
 $$
   \mathbf{u}_t(\mathbf{x}) = \boldsymbol{\mu}_1-\boldsymbol{\mu}_0 + \frac{\dot{\sigma}_t^2}{2\sigma_t^2} (\mathbf{x} - \boldsymbol{\mu}_t).
   $$
   Further simplification yields:
   $$
   \boxed{\mathbf{u}_t(\mathbf{x}) = \frac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{\sigma_t^2}, \quad \sigma_t^2 = 2t^2 - 2t + 1.}
   $$
Check
	- $t=0$,   $\mathbf{u}_0(\mathbf{x}) = \boldsymbol{\mu} - \mathbf{x}_0 = \mathbb{E}(\mathbf{x}_1)-\mathbf{x}_0$   (因為 independent, 取 $x_1$ 平均作為 vector 終點)
	- $t=1$,   $\mathbf{u}_1(\mathbf{x}) = \boldsymbol{\mu} + \mathbf{x}_1 = -(\mathbb{E}(\mathbf{x}_0)-\mathbf{x}_1)$   (基本是上面的反向，所以多一個負號）

### Final Closed-Form:
With $\boldsymbol{\mu} = [10, 0]$, the vector field is:
$$
\mathbf{u}_t(\mathbf{x}) = \frac{1}{2t^2 - 2t + 1} \begin{pmatrix} (2t-1)x_1 + 10 \\ (2t-1)x_2 \end{pmatrix},
$$
where $\mathbf{x} = [x_1, x_2]^\top$.

#### Verification:
- At $t=0$: $\mathbf{u}_0(\mathbf{x}) = [-x_1 + 10, -x_2]^\top$, which transports $\mathcal{N}([-10, 0]^\top, \mathbf{I})$ as expected.
- At $t=1$: $\mathbf{u}_1(\mathbf{x}) = [x_1 + 10, x_2]^\top$, which transports to $\mathcal{N}([10, 0]^\top, \mathbf{I})$.
- The continuity equation $\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \mathbf{u}_t) = 0$ holds for the Gaussian path.

$$
\boxed{\mathbf{u}_{t}(\mathbf{x}) = \dfrac{1}{2t^{2} - 2t + 1} \begin{pmatrix} (2t - 1) x_{1} + 10 \\ (2t - 1) x_{2} \end{pmatrix}}
$$

#### Validate the conservation of probability (Appendix E)!
$$
\boxed{\frac{d}{dt} \log p_t(\mathbf{x}) = - \nabla \cdot \mathbf{u}_t(\mathbf{x}) = -d \frac{2t-1}{\sigma_t^2} = - \frac{\mathbf{d}\dot{\sigma_t^2}}{2\sigma_t^2}=-\frac{\mathbf{d}}{2}\frac{ d}{dt}\log\sigma_t^2}
$$
最後的
t < 0.5  d/dt > 0, 代表這個 flow 是壓縮。
t > 0.5  d/dt < 0, 代表這個 flow 是膨脹。


![[Pasted image 20250604104124.png]]

Next, I'll derive the closed-form vector field using the Conditional Flow Matching (CFM) definition from Lipman et al. (2023), following the formula you provided:

## Method 2: Derivation using CFM Framework

Given:
- **Prior distribution** $p_0 = \mathcal{N}(-\boldsymbol{\mu}, \mathbf{I})$
- **Target distribution** $p_1 = \mathcal{N}(+\boldsymbol{\mu}, \mathbf{I})$
- **Conditional vector field** $u_t(x \mid x_1) = \frac{x_1 - x}{1-t}$ (linear interpolation path)
- **Conditional probability path** $p_t(x \mid x_1) = \mathcal{N}(x; \; tx_1 - (1-t)\boldsymbol{\mu}, \;(1-t)^2\mathbf{I})$

The marginal vector field is:
$$
u_t(x) = \mathbb{E}_{x_1 \sim p_{1|t}} \left[ u_t(x \mid x_1) \right] = \int u_t(x \mid x_1) \frac{p_t(x \mid x_1) p_1(x_1)}{p_t(x)} dx_1
$$

#### **Step 1: Identify Components**
1. $p_1(x_1) = \mathcal{N}(x_1; \boldsymbol{\mu}, \mathbf{I})$
2. $p_t(x \mid x_1) = \mathcal{N}(x; tx_1 - (1-t)\boldsymbol{\mu}, (1-t)^2\mathbf{I})$
3. $p_t(x) = \mathcal{N}(x; (2t-1)\boldsymbol{\mu}, (2t^2-2t+1)\mathbf{I})= \mathcal{N}(x; \boldsymbol{\mu}_t, \sigma^2_t \mathbf{I})$ (marginal distribution) 
	-  $\boldsymbol{\mu}_t = (2t-1)\boldsymbol{\mu} , \,\,\sigma^2_t = 2t^2-2t+1$

The posterior is Gaussian:
$$\begin{align}
p_{1|t}(x_1|x) &= \mathcal{N}\left( x_1; \frac{tx + (1-t)\boldsymbol{\mu}}{2t^2-2t+1}, \frac{(1-t)^2}{2t^2-2t+1} \mathbf{I}\right)\\
&= \mathcal{N}\left( x_1; \frac{tx + (1-t)\boldsymbol{\mu}}{\sigma^2_t}, \frac{(1-t)^2}{\sigma^2_t} \mathbf{I}\right)\\
\end{align}$$

#### **Step 2: Compute Expectation** (Appendix C)

The conditional vector field should be: 
$$
\begin{aligned}
u_t(x) & =\mathbb{E}_{x_1 \sim p_{1 \mid t}}\left[u_t\left(x \mid x_1\right)\right] \\
& =\int u_t\left(x \mid x_1\right) \frac{p_t\left(x \mid x_1\right) q_1\left(x_1\right)}{p_t(x)} \mathrm{d} x_1 .
\end{aligned}
$$

因爲 Condition flow 是 Gaussian given $\mathbf{x}_1$
$$\begin{aligned}
u_t(x\mid x_1) &= u(x_t\mid x_1) = \frac{\psi_t(x_0)}{dt} = \frac{d x_t}{dt} = \dot{\sigma}_t(x_1) x_0 + \dot{\mu_t}(x_1)\\
&= \dot{\sigma}_t(x_1) \left[\frac{x_t - u_t(x_1)}{\sigma_t(x_1)}\right] + \dot{\mu_t}(x_1) \\
&= \frac{\dot{\sigma}_t(x_1)}{\sigma_t(x_1)} (x - \mu_t(x_1))+ \dot{\mu_t}(x_1) \\
\end{aligned}$$
但是 condition Gaussian flow on $x_1$ 和 uncondition Gaussian flow 不同。是一個像錐體如下圖。
![[Pasted image 20250514121948.png]]

Uncondition Gaussian flow 則是像束腰的圓柱體。

![[Pasted image 20250604104124.png]]

所以雖然都是
$$\boxed{\mathbf{u}_t(\mathbf{x}) = \dot{\boldsymbol{\mu}}_t + \frac{\dot{\sigma_t}}{\sigma_t} (\mathbf{x} - \boldsymbol{\mu}_t)}$$
#### **Condition Vector Field:  $u_t(\mathbf{x} \mid \mathbf{x}_1)$ 

给定终端值 $\mathbf{x}_1$，条件路径为：
$$
\mathbf{x}_t \mid \mathbf{x}_1 = (1-t)\mathbf{x}_0 + t\mathbf{x}_1
$$
条件分布为高斯：
$$
\mathbf{x}_t \mid \mathbf{x}_1 \sim \mathcal{N}\left( \boldsymbol{\mu}_t(\mathbf{x}_1), \sigma_t^2(\mathbf{x}_1) \mathbf{I} \right)
$$
其中：
- $\boldsymbol{\mu}_t(\mathbf{x}_1) = (1-t)\boldsymbol{\mu}_0 + t\mathbf{x}_1$  因爲最後收斂到 $x_1$
- $\sigma_t(\mathbf{x}_1) = (1-t)\sigma_0$（因为方差为 $(1-t)^2 \sigma_0^2$，标准差为 $|1-t|\sigma_0$，且 $t \in [0,1]$ 时 $1-t \geq 0$)

条件向量场由路径的导数给出：
$$\begin{align}
\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) &=  \dot{\boldsymbol{\mu}}_t(\mathbf{x}_1) + \frac{\dot{\sigma_t}(\mathbf{x}_1)}{\sigma_t(\mathbf{x}_1)} (\mathbf{x}_t - \boldsymbol{\mu}_t(\mathbf{x}_1)) \\
&= \mathbf{x}_1 - \boldsymbol{\mu}_0 + \frac{-\sigma_0}{(1-t)\sigma_0}(\mathbf{x}_t-\boldsymbol{\mu}_t(\mathbf{x}_1)) \\
&=\frac{(\mathbf{x}_1 - \boldsymbol{\mu}_0)(1-t)}{1-t} + \frac{-\mathbf{x}_t+(1-t)\boldsymbol{\mu}_0 + t\mathbf{x}_1}{1-t}\\
&=\frac{\mathbf{x}_1 -\mathbf{x}_t}{1-t} \\
\end{align}$$
另一個方法是直接用直綫假説
$$\begin{align}
\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) &= -\mathbf{x}_0 + \mathbf{x}_1
\end{align}$$
代入 $\mathbf{x}_0 = \frac{\mathbf{x}_t - t\mathbf{x}_1}{1-t}$：
$$
\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) = -\frac{\mathbf{x}_t - t \mathbf{x}_1}{1-t} + \frac{\mathbf{x}_1 -t\mathbf{x}_1} {1-t} =  \frac{\mathbf{x}_1 -\mathbf{x}_t} {1-t}
$$

#### **Uncondition Vector Field: $u_t(\mathbf{x})$
无条件向量场是条件向量场关于后验分布 $p(\mathbf{x}_1 \mid \mathbf{x}_t)$ 的期望：
$$
\mathbf{u}_t(\mathbf{x}) = \mathbb{E}_{\mathbf{x}_1 \sim p(\mathbf{x}_1 \mid \mathbf{x}_t)} \left[ \mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) \right]
$$


Since $\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) = \frac{\mathbf{x}_1 - \mathbf{x}}{1-t}$ is linear in $\mathbf{x}_1$, and the posterior is Gaussian:
$$
\mathbf{u}_t(x) = \mathbb{E}_{\mathbf{x}_1 \sim p_{1\mid t}(\mathbf{x}_1|\mathbf{x})}\left[\frac{\mathbf{x}_1 - \mathbf{x}}{1-t} \right] = \frac{\overbrace{\mathbb{E}_{p_{1\mid t}}[\mathbf{x}_1 ]}^{\text{posterior mean}} - \mathbf{x}}{1-t}
$$



由于联合分布 $(\mathbf{x}_t, \mathbf{x}_1)$ 是高斯分布，后验 $p(\mathbf{x}_1 \mid \mathbf{x}_t)$ 也是高斯分布。计算其均值和协方差：
- 联合分布：
  $$
  \begin{bmatrix} \mathbf{x}_t \\ \mathbf{x}_1 \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \boldsymbol{\mu}_t \\ \boldsymbol{\mu}_1 \end{bmatrix}, \begin{bmatrix} \sigma_t^2 \mathbf{I} & t\sigma_1^2 \mathbf{I} \\ t\sigma_1^2 \mathbf{I} & \sigma_1^2 \mathbf{I} \end{bmatrix} \right)
  $$
- 后验均值：
  $$
  \mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t] = \boldsymbol{\mu}_1 + \frac{t\sigma_1^2}{\sigma_t^2} (\mathbf{x} - \boldsymbol{\mu}_t) = \boldsymbol{\mu} + \frac{t(\mathbf{x} - (2t-1)\boldsymbol{\mu})}{2t^2-2t+1} 
  $$

代入期望：
$$
\mathbf{u}_t(\mathbf{x}) = \left[\boldsymbol{\mu} + \frac{t(\mathbf{x} - (2t-1)\boldsymbol{\mu})}{2t^2-2t+1} - \mathbf{x}\right]/(1-t)=\frac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{2t^2-2t+1}$$

$$
\boxed{\mathbf{u}_t(\mathbf{x}) = \frac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{2t^2-2t+1} = \frac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{\sigma^2_t}}
$$

## Uncondition Flow Geometric Interpretation
看下圖比較清楚：想像是一團 centered at (-10, 0) 的點，隨著時間往 (+10, 0) 移動的過程。綠色是 trace, 是每個點經過 vector field 被改變之後的 trace.  可以想像是微分方程的解。

$\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{u}_t(\boldsymbol{x_t})$

- **For $t < 0.5$:**  
  The term $(2t-1)$ is negative, 所以是一個壓縮的流。
  
- **For $t > 0.5$:**  
  The term $(2t-1)$ is positive, 所以是一個膨脹的流。

![[Pasted image 20250604104124.png]]

## 例二：General G2G with Scaled Indentity Covariance (Appendix E)

Under independent coupling:
- $\mathbf{x}_0 \sim \mathcal{N}(\boldsymbol{\mu}_0, \sigma_0^2 \mathbf{I})$
- $\mathbf{x}_1 \sim \mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2 \mathbf{I})$

With linear interpolation $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$, the marginal is Gaussian:
$$
\mathbf{x}_t \sim \mathcal{N}(\boldsymbol{\mu}_t, \sigma_t^2 \mathbf{I}),
$$
where:
- **Mean**: $\boldsymbol{\mu}_t = (1-t)\boldsymbol{\mu}_0 + t\boldsymbol{\mu}_1$
- **Variance**: $\sigma_t^2 = (1-t)^2 \sigma_0^2 + t^2 \sigma_1^2$


The form below **holds generally for isotropic Gaussians** $p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \sigma_0^2 \mathbf{I})$ and $p_1 = \mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2 \mathbf{I})$ under independent coupling.  而且下式是座標無關形式！

$$\boxed{\mathbf{u}_t(\mathbf{x}) = \dot{\boldsymbol{\mu}}_t + \frac{\dot{\sigma_t^{2}}}{2\sigma_t^{2}} (\mathbf{x} - \boldsymbol{\mu}_t)}$$
補充一下，因爲 $\dot{\sigma_t^{2}}=2 \sigma_t \dot{\sigma_t}$.  所以上式也可以寫成：
$$\mathbf{u}_t(\mathbf{x}) = \dot{\boldsymbol{\mu}}_t + \frac{\dot{\sigma_t^{2}}}{2\sigma_t^{2}} (\mathbf{x} - \boldsymbol{\mu}_t)= \dot{\boldsymbol{\mu}}_t + \frac{\dot{\sigma_t}}{\sigma_t} (\mathbf{x} - \boldsymbol{\mu}_t)$$
看起來更簡潔和直觀，但是對於 general form 會有一點 messy, 所以我們 keep both forms.

例一：  $\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0 = 2\boldsymbol{\mu}$,  ${\sigma}^2_t = 2t^2-2t+1$, $\dot{\sigma^2_t} = 2(2t-1)$, $\boldsymbol{\mu}_t = (2t-1)\boldsymbol{\mu}$, $\mathbf{d}=2$ (dimension), 帶入得到
$$
\boxed{\mathbf{u}_t(\mathbf{x}) = \frac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{2t^2-2t+1} = \frac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{\sigma^2_t}}
$$

同樣 conservation of probability 如下，也是座標無關形式！
$$
\boxed{\frac{d}{dt} \log p_t(\mathbf{x}) = - \nabla \cdot \mathbf{u}_t(\mathbf{x}) = -\frac{\mathbf{d}}{2}\frac{ d}{dt}\log\sigma_t^2=-\mathbf{d}\frac{ d}{dt}\log\sigma_t}
$$

## 例三：General G2G with Full Rank Covariance (Appendix G)

The vector field $u_t(x)$ for flow matching between two Gaussian distributions $p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$ and $p_1 = \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ is derived under a **Gaussian probability path** where the mean and covariance are linearly interpolated:

$$
\boldsymbol{\mu}_t = (1 - t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1
$$
$$
\boldsymbol{\Sigma}_t = (1 - t)^2 \boldsymbol{\Sigma}_0 + t^2 \boldsymbol{\Sigma}_1
$$

The vector field is given by:

$$
\mathbf{u}_t(x) = \underbrace{\dot{\boldsymbol{\mu}}_t}_{\text{Mean component}\,} + \underbrace{\frac{1}{2} \dot{\boldsymbol{\Sigma}_t} \boldsymbol{\Sigma}_t^{-1} ((\mathbf{x} - \boldsymbol{\mu}_t)}_{\text{Covariance component}}
$$
$$
\boxed{\mathbf{u}_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \dfrac{ t \boldsymbol{\Sigma}_1 - (1-t) \boldsymbol{\Sigma}_0 }{ \boldsymbol{\Sigma}_t } (\mathbf{x} - \boldsymbol{\mu}_t)}
$$

If $\boldsymbol{\Sigma}_0$ and $\boldsymbol{\Sigma}_1$ **commute**, the flow simplifies to: (Appendix H)
$$
\boxed{\mathbf{x}(t) = \boldsymbol{\mu}_t + \boldsymbol{\Sigma}_t^{1/2} \boldsymbol{\Sigma}_0^{-1/2} \left( \mathbf{x}_0 - \boldsymbol{\mu}_0 \right)}
$$

## 例三：Full-Rank Gaussian-to-Low-Rank Gaussian (Appendix E)

![[Pasted image 20250607094307.png]]
我們假設 $p_1 \sim N(\mu_1, \Sigma_{min})$ and  $p_0 \sim N(\mu_0, \Sigma_{max})$ 

$$
x + 1 \over \sqrt{1 - x^2} \tag{2}
$$
$$
E = mc^2
$$
^energy-eq

As shown in equation [[#^energy-eq]], ...



假設 $X_1 \in$ {”貓", "狗“} 的機率為 10%, 90%.  
u(x_t)  在 t =0 是各 50%, 50%.  但是到 t = 1 是 10%, 90%,  那在 t 中間如何?

u_t = u(x_t | x_1="貓") * 


### Why Conditional Flow Matching?

Why conditional vector field?  因為 flow matching 是 sampling from $p_t(x)$，但是 conditional flow matching 可以從 data $q(x_1)$ sampling 來 training.  
- 但是還是有 $p_t(x\mid x_1)$ 才能 training?  如同 diffusion 的 transition probability: 老把戲是 Gaussian. $p_t(x\mid x_1)\sim N(\mu_t(x_1), \sigma^2_t(x_1) I)$
- 這個 Gaussian equivalently!  $x_{t\mid 1} = \mu_t(x_1) + \sigma_t(x_1) \cdot z, \quad z\sim N(0, I)$
- 因為 $z$ 和 $x_0$ 一樣 $N(0, I)$，也可以改成：$x_{t\mid 1} = \mu_t(x_1) + \sigma_t(x_1)  x_0$
- 再因為 $x_{t\mid 1} = \phi(x_0\mid x_1)=\psi(x_0)$,  所以也可以寫：$\psi(x_0) = \mu_t(x_1) + \sigma_t(x_1)  x_0$

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
**Condition vector field 的形式和 uncondition vector field 一樣！但是是 Condition vector field 是 given $x_1$,  所以值完全不同！**

> [!NOTE]
> 如果是 OT, $u_t(x\mid x_1)$ 就是直綫的斜率:  
> Since $u_t(x \mid x_1) = \frac{x_1 - x}{1-t}$ is linear in $x_1$, and the posterior is Gaussian:


**上式的好處是如果我們已經知道 $p_t$ 在時間和空間的分佈, i.e. $p(x, t)$ from Fokker-Planck equation，可以直接轉換成 flow!!**
$$p_t(x\mid x_1)\sim N(\mu_t(x_1), \sigma^2_t(x_1) I)$$


這個 conditional flow 用圖看比較清楚。
最後 reach $x_1=X_1$, 從一個 fat initial condition ($\sigma_t(x_1)$ 隨時間變小)，但是最終收斂到 $\mu_t(x_1)=X_1$
![[Pasted image 20250514121948.png]]
#### 如何 sample：$\psi(x_0) = \mu_t(x_1) + \sigma_t(x_1)  x_0$ and $\frac{d}{dt}\psi_t(x_0)$?

$t \sim [0, 1]$.
$x_1$ 是直接從 data set sample 的 image $q(x_1)$.
$x_0 \sim N(0, I)$ 也非常簡單。


## Sampling (from $x_0$ and $u_t$ to get $x_t$)

**最重要的是 $x_t$不是直線**！因為 $\mathbf{u}_t(\mathbf{x}_t)$ 是一個平均的結果，不是一個 constant vector!
但是 condition vector 是直線 (in the OT case).   

理論上非常簡單，就是解一個 ODE with initial condition $\mathbf{x}_0$ is:
$$\frac{d\mathbf{x}}{dt} = \mathbf{u}_t(\mathbf{x})$$ The exact solution for 例一 of the ODE 
$$
\mathbf{x}_t = (2t-1)\boldsymbol{\mu} + \sqrt{2t^2 - 2t + 1} \cdot (\mathbf{x}_0 + \boldsymbol{\mu})
$$
- **At $t=0$**: $\mathbf{x}_0 = -\boldsymbol{\mu} + \mathbf{z}$ (where $\mathbf{z} = \mathbf{x}_0 + \boldsymbol{\mu} \sim \mathcal{N}(0, \mathbf{I})$).  
- **At $t=1$**: $\mathbf{x}_1 = \boldsymbol{\mu} + \mathbf{z} = \mathbf{x}_0 + 2\boldsymbol{\mu}$ (a sample from $p_1$).  

因爲 flow $\phi_t(\mathbf{x}_0) = \mathbf{x}_t$
$$
\phi_t(\mathbf{x}_0) =\mathbf{x}_t = (2t-1)\boldsymbol{\mu} + \sqrt{2t^2 - 2t + 1} \cdot (\mathbf{x}_0 + \boldsymbol{\mu})
$$
The path (flow $\phi(x_0)$) is **curved** due to the $\sqrt{2t^2 - 2t + 1}$ term, which is nonlinear in $t$.

通用的表示：（Two indepedent Gaussians, Appendix E and F)
$$\boxed{\mathbf{u}_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \frac{ {d\log\sigma}_t^{2}}{2 dt} (\mathbf{x} - \boldsymbol{\mu}_t)}$$
$$\begin{aligned}
\mathbf{x}(t) &= \left[(1-t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1\right] + \dfrac{\sqrt{(1-t)^2 \sigma_0^2 + t^2 \sigma_1^2}}{\sigma_0} (\mathbf{x}_0 - \boldsymbol{\mu}_0)\\
\phi_t(\mathbf{x}_0) &= \mathbf{x}(t)=\boldsymbol{\mu}_t + \dfrac{\sigma_t}{\sigma_0} (\mathbf{x}_0 - \boldsymbol{\mu}_0)\\
\end{aligned}
$$
- $\boldsymbol{\mu}_t = (1-t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1 = \boldsymbol{\mu}_0 + t(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)$ 是以 $\boldsymbol{\mu}_0$ 為起點，斜率為 $(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)$ 的直綫。只有在$\mathbf{x}_0 = \boldsymbol{\mu}_0$ 才會走這條直綫。當 $\mathbf{x}_0 \ne \boldsymbol{\mu}_0$ 偏離的部分就會照 standard deviation 比例
 ($\frac{\sigma_t}{\sigma_0}$) 加到這條直綫。 
- $t =0, \phi_0(x_0) = x_0$
- $t =1, \phi_1(x_0) = \mu_1 + \frac{\sigma_1}{\sigma_0}(x_0-\mu_0)$.   如果 $\sigma_1=\sigma_0=\sigma$ , $\phi_1(x_0) = x_0+(\mu_1-\mu_0)$.  即是所有的終點都是起點加上 mean difference.

#### General Expression
$\boldsymbol{\mu}_t = (1 - t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1$ and $\boldsymbol{\Sigma}_t = (1 - t)^2 \boldsymbol{\Sigma}_0 + t^2 \boldsymbol{\Sigma}_1$
If $\boldsymbol{\Sigma}_0$ and $\boldsymbol{\Sigma}_1$ **commute**, the flow simplifies to: (Appendix H)
$$
\boxed{\phi_t(\mathbf{x}_0)=\mathbf{x}(t) = \boldsymbol{\mu}_t + \boldsymbol{\Sigma}_t^{1/2} \boldsymbol{\Sigma}_0^{-1/2} \left( \mathbf{x}_0 - \boldsymbol{\mu}_0 \right)}
$$



### Example: $\mathbf{x}_0 = [-10, 1]^\top$
- **Target**: $\mathbf{x}_1 = \mathbf{x}_0 + 2\boldsymbol{\mu} = [10, 1]^\top$.  
- **Trajectory**:  
  $$
  \mathbf{x}_t = \begin{bmatrix} 20t - 10 \\ \sqrt{2t^2 - 2t + 1} \end{bmatrix}
  $$
- **Positions**:  
  - $t=0$: $[-10, 1]^\top$  
  - $t=0.5$: $[0, \sqrt{0.5}]^\top \approx [0, 0.707]^\top$  
  - $t=1$: $[10, 1]^\top$.  

### Why Straight Lines Do *Not* Occur:
1. **Independent coupling**:  
   - For a fixed pair $(\mathbf{x}_0, \mathbf{x}_1)$, the conditional path **is straight**: $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$.  
   - However, $\mathbf{u}_t(\mathbf{x})$ is the *marginal* field (average over all $\mathbf{x}_1$), so individual paths **curve** to reconcile all possible endpoints.
   
2. **Geometry**:  
   - Straight lines require $\frac{d^2\mathbf{x}_t}{dt^2} = 0$. Here, acceleration is nonzero:  
     $$
     \frac{d\mathbf{u}_t}{dt} \neq 0 \implies \text{curved paths}.
     $$




### Conclusion:
**Wrong.** Under the given marginal vector field $\mathbf{u}_t(\mathbf{x})$:  
- Samples from $p_0$ travel to $\mathbf{x}_0 + 2\boldsymbol{\mu}$ (a valid sample from $p_1$).  
- The trajectory is **not a straight line** unless $\mathbf{x}_0 = -\boldsymbol{\mu}$ (mean of $p_0$).  
- Curved paths arise from the independent coupling, where the vector field averages over all possible $\mathbf{x}_1$.

> **Key Takeaway**: The marginal flow matches the distributions $p_0 \to p_1$ but follows *curved trajectories*. For straight lines, use **conditional flow matching** (Lipman et al.) with paired samples $(\mathbf{x}_0, \mathbf{x}_1)$.




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

**對應的 condition vector field，物理意義非常簡單，就是一個 constant field 和 sampled 的 $t$ 無關！而且是 $x_0$ 和目標的$x_1$ 的向量差！就是一路直衝終點！** Wrong, 我們不知道 $x_1$!
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




## Reference

MIT 6.S184: Flow Matching and Diffusion Models https://www.youtube.com/watch?v=GCoP2w-Cqtg&t=28s&ab_channel=PeterHolderrieth

Yaron Meta paper: [[2210.02747] Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

An Introduction to Flow Matching: https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html


## Appendix C

Compute Expectation** (Appendix C)

The conditional vector field should be: 
![[Pasted image 20250603200549.png]]

Since $u_t(x \mid x_1) = \frac{x_1 - x}{1-t}$ is linear in $x_1$, and the posterior is Gaussian:
$$
u_t(x) = \mathbb{E}_{x_1 \sim p_{1\mid t}(x_1|x)}\left[\frac{x_1 - x}{1-t} \right] = \frac{\overbrace{\mathbb{E}_{p_{1\mid t}}[x_1 ]}^{\text{posterior mean}} - x}{1-t}
$$
$$
\boxed{\mathbf{u}_t(\mathbf{x}) = \frac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{2t^2-2t+1} = \frac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{\sigma^2_t}}
$$
### **Geometric Interpretation**
看下圖比較清楚：想像是一團 centered at (-10, 0) 的點，隨著時間往 (+10, 0) 移動的過程。綠色是 trace, 是每個點經過 vector field 被改變之後的 trace.  可以想像是微分方程的解。

$\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{u}_t(\boldsymbol{x_t})$

- **For $t < 0.5$:**  
  The term $(2t-1)$ is negative, 所以是一個壓縮的流。
  
- **For $t > 0.5$:**  
  The term $(2t-1)$ is positive, 所以是一個膨脹的流。

![[Pasted image 20250604104124.png]]

The posterior mean is:
$$
\mathbb{E}[x_1 \mid x_t = x] = \frac{tx + (1-t)\boldsymbol{\mu}}{2t^2-2t+1}
$$

#### **Step 3: Substitute and Simplify**
$$
u_t(x) = \frac{1}{1-t} \left( \frac{tx + (1-t)\boldsymbol{\mu}}{2t^2-2t+1} - x \right)
$$

$$
= \frac{1}{1-t} \left( \frac{t x + (1-t)\boldsymbol{\mu} - x(2t^2-2t+1)}{2t^2-2t+1} \right)
$$

$$
= \frac{1}{1-t} \cdot \frac{(t - 2t^2 + 2t - 1)x + (1-t)\boldsymbol{\mu}}{2t^2-2t+1}
$$

$$
= \frac{1}{1-t} \cdot \frac{(3t - 2t^2 - 1)x + (1-t)\boldsymbol{\mu}}{2t^2-2t+1}
$$
$$
\boxed{\mathbf{u}_t(\mathbf{x}) = \frac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{2t^2-2t+1}}
$$

### **Verification**.  
1. **At $t = 0$:**
$$
\mathbf{u}_0(\mathbf{x}) = \frac{(-1)\mathbf{x} + \boldsymbol{\mu}}{1} = -\mathbf{x}+\boldsymbol{\mu}
$$
- At prior mean $\mathbf{x} = -\boldsymbol{\mu}$: $\mathbf{u}_0(-\boldsymbol{\mu}) = -(-\boldsymbol{\mu})+\boldsymbol{\mu} = 2\boldsymbol{\mu}$  
  (Points toward $+2\boldsymbol{\mu}$, correct)

1. **At $t = 1$:**
$$
\mathbf{u}_1(\mathbf{x}) = \frac{(2-1)\mathbf{x} + \boldsymbol{\mu}}{2-2+1} = \mathbf{x} + \boldsymbol{\mu}
$$
- At target mean $\mathbf{x} = \boldsymbol{\mu}$: $\mathbf{u}_1(\boldsymbol{\mu}) = \boldsymbol{\mu} + \boldsymbol{\mu} = 2\boldsymbol{\mu}$  
  (Consistent with linear interpolation)

### **Implementation**
```python
import numpy as np

def vector_field(x: np.ndarray, t: float, mu: np.ndarray) -> np.ndarray:
    """
    Computes CFM vector field for p0 = N(-μ, I) → p1 = N(+μ, I).
    
    Args:
        x: Current position (n-dimensional vector)
        t: Time in [0, 1]
        mu: Target mean vector (+μ)
    
    Returns:
        u_t(x): Vector field direction
    """
    numerator = (2*t - 1) * x + mu
    denominator = 2*t**2 - 2*t + 1
    return numerator / (denominator + 1e-8)  # Avoid division by zero
```






其實這也是 Schrödinger Bridge.

![[Pasted image 20250602232444.png]]


One-sided flow matching?  Two sided flow matching?


## Appendix D

To derive the conditional expectation $\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}]$ for the linear interpolation path $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$, where $\mathbf{x}_0 \sim \mathcal{N}(-\boldsymbol{\mu}, \mathbf{I})$ and $\mathbf{x}_1 \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{I})$ are independent, follow these steps:

### **Step 1: Define Joint Distribution**
The vector $\begin{bmatrix} \mathbf{x}_0 \\ \mathbf{x}_t \end{bmatrix}$ is jointly Gaussian since $\mathbf{x}_t$ is a linear combination of $\mathbf{x}_0$ and $\mathbf{x}_1$. Compute its moments:  
- **Means**:  
  $$
  \mathbb{E}[\mathbf{x}_0] = -\boldsymbol{\mu}, \quad \mathbb{E}[\mathbf{x}_t] = (1-t)(-\boldsymbol{\mu}) + t(\boldsymbol{\mu}) = (2t-1)\boldsymbol{\mu}.
  $$  
- **Covariances**:  
  $$
  \text{Cov}(\mathbf{x}_0) = \mathbf{I}, \quad \text{Cov}(\mathbf{x}_t) = (1-t)^2\mathbf{I} + t^2\mathbf{I} = \sigma_t^2 \mathbf{I}, \quad \sigma_t^2 = 2t^2 - 2t + 1.
  $$  
- **Cross-Covariance**:  
  $$
  \text{Cov}(\mathbf{x}_0, \mathbf{x}_t) = \mathbb{E}[(\mathbf{x}_0 + \boldsymbol{\mu})(\mathbf{x}_t - (2t-1)\boldsymbol{\mu})^\top] = (1-t)\mathbf{I},
  $$  
  since $\mathbf{x}_1 - \boldsymbol{\mu}$ is independent of $\mathbf{x}_0 + \boldsymbol{\mu}$ and has zero mean.

### **Step 2: Apply Gaussian Conditioning Formula**
For jointly Gaussian vectors $\begin{bmatrix} \mathbf{a} \\ \mathbf{b} \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \boldsymbol{\mu}_a \\ \boldsymbol{\mu}_b \end{bmatrix}, \begin{bmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end{bmatrix} \right)$,  
$$
\mathbb{E}[\mathbf{a} \mid \mathbf{b} = \mathbf{x}] = \boldsymbol{\mu}_a + \Sigma_{ab} \Sigma_{bb}^{-1} (\mathbf{x} - \boldsymbol{\mu}_b).
$$  
Here, $\mathbf{a} = \mathbf{x}_0$, $\mathbf{b} = \mathbf{x}_t$, and:  
$$
\boldsymbol{\mu}_a = -\boldsymbol{\mu}, \quad \boldsymbol{\mu}_b = (2t-1)\boldsymbol{\mu}, \quad \Sigma_{ab} = (1-t)\mathbf{I}, \quad \Sigma_{bb} = \sigma_t^2 \mathbf{I}.
$$

### **Step 3: Substitute and Simplify**
$$
\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}] = -\boldsymbol{\mu} + \left[(1-t)\mathbf{I}\right] \left[\sigma_t^2 \mathbf{I}\right]^{-1} (\mathbf{x} - (2t-1)\boldsymbol{\mu}).
$$  
Since $\left[\sigma_t^2 \mathbf{I}\right]^{-1} = \frac{1}{\sigma_t^2} \mathbf{I}$:  
$$
= -\boldsymbol{\mu} + \frac{1-t}{\sigma_t^2} (\mathbf{x} - (2t-1)\boldsymbol{\mu}).
$$

### **Final Result**
$$
\boxed{\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}] = -\boldsymbol{\mu} + \dfrac{1-t}{\sigma_t^{2}} \left( \mathbf{x} - (2t-1)\boldsymbol{\mu} \right)}
$$

### **Intuition**  
The term $\frac{1-t}{\sigma_t^2}$ represents the **regression coefficient** adjusting for the correlation between $\mathbf{x}_0$ and $\mathbf{x}_t$. The expression linearly combines the prior mean $-\boldsymbol{\mu}$ with the deviation of $\mathbf{x}$ from the marginal mean $(2t-1)\boldsymbol{\mu}$, scaled by the relative variance contribution of $\mathbf{x}_0$ to $\mathbf{x}_t$.


## Appendix E

The vector field is given by:
$$
\mathbf{u}_t(\mathbf{x}) = \frac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{2t^2 - 2t + 1}, \quad \boldsymbol{\mu} = [10, 0]^\top
$$

This vector field defines the **marginal flow** (not conditional paths). To determine whether samples follow straight lines from $\mathbf{x}_0$ to $\mathbf{x}_1$, we analyze the trajectory under this field.

### Key Insight:
- The marginal vector field $\mathbf{u}_t(\mathbf{x})$ is derived from the **independent coupling** of $p_0$ and $p_1$, where $\mathbf{x}_0 \sim \mathcal{N}(-\boldsymbol{\mu}, \mathbf{I})$ and $\mathbf{x}_1 \sim \mathcal{N}(+\boldsymbol{\mu}, \mathbf{I})$ are sampled *independently*.
- For a fixed $\mathbf{x}_0$, the endpoint $\mathbf{x}_1$ is **not unique** (since $\mathbf{x}_1$ is random and independent of $\mathbf{x}_0$).
- The flow under $\mathbf{u}_t(\mathbf{x})$ transports $\mathbf{x}_0$ to $\mathbf{x}_0 + 2\boldsymbol{\mu}$ (a sample from $p_1$), but the path is **not straight** in general.

### Trajectory Analysis:
The exact solution of the ODE $\frac{d\mathbf{x}_t}{dt} = \mathbf{u}_t(\mathbf{x}_t)$ with initial condition $\mathbf{x}_0$ is:
$$
\mathbf{x}_t = (2t-1)\boldsymbol{\mu} + \sqrt{2t^2 - 2t + 1} \cdot (\mathbf{x}_0 + \boldsymbol{\mu})
$$
- **At $t=0$**: $\mathbf{x}_0 = -\boldsymbol{\mu} + \mathbf{z}$ (where $\mathbf{z} = \mathbf{x}_0 + \boldsymbol{\mu} \sim \mathcal{N}(0, \mathbf{I})$).  
- **At $t=1$**: $\mathbf{x}_1 = \boldsymbol{\mu} + \mathbf{z} = \mathbf{x}_0 + 2\boldsymbol{\mu}$ (a sample from $p_1$).  

The path is **curved** due to the $\sqrt{2t^2 - 2t + 1}$ term, which is nonlinear in $t$.

### Example: $\mathbf{x}_0 = [-10, 1]^\top$
- **Target**: $\mathbf{x}_1 = \mathbf{x}_0 + 2\boldsymbol{\mu} = [10, 1]^\top$.  
- **Trajectory**:  
  $$
  \mathbf{x}_t = \begin{bmatrix} 20t - 10 \\ \sqrt{2t^2 - 2t + 1} \end{bmatrix}
  $$
- **Positions**:  
  - $t=0$: $[-10, 1]^\top$  
  - $t=0.5$: $[0, \sqrt{0.5}]^\top \approx [0, 0.707]^\top$  
  - $t=1$: $[10, 1]^\top$.  

The $y$-component dips to $\approx 0.707$ at $t=0.5$ (not a straight line to $[10, 1]^\top$):  
### Why Straight Lines Do *Not* Occur:
1. **Independent coupling**:  
   - For a fixed pair $(\mathbf{x}_0, \mathbf{x}_1)$, the conditional path **is straight**: $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$.  
   - However, $\mathbf{u}_t(\mathbf{x})$ is the *marginal* field (average over all $\mathbf{x}_1$), so individual paths **curve** to reconcile all possible endpoints.
   
2. **Geometry**:  
   - Straight lines require $\frac{d^2\mathbf{x}_t}{dt^2} = 0$. Here, acceleration is nonzero:  
     $$
     \frac{d\mathbf{u}_t}{dt} \neq 0 \implies \text{curved paths}.
     $$

### Conclusion:
**Wrong.** Under the given marginal vector field $\mathbf{u}_t(\mathbf{x})$:  
- Samples from $p_0$ travel to $\mathbf{x}_0 + 2\boldsymbol{\mu}$ (a valid sample from $p_1$).  
- The trajectory is **not a straight line** unless $\mathbf{x}_0 = -\boldsymbol{\mu}$ (mean of $p_0$).  
- Curved paths arise from the independent coupling, where the vector field averages over all possible $\mathbf{x}_1$.

> **Key Takeaway**: The marginal flow matches the distributions $p_0 \to p_1$ but follows *curved trajectories*. For straight lines, use **conditional flow matching** (Lipman et al.) with paired samples $(\mathbf{x}_0, \mathbf{x}_1)$.


## Appendix F

To compute the total derivative of $\log p_t(\mathbf{x})$ along the probability flow defined by the vector field $\mathbf{u}_t(\mathbf{x})$, we use the formula:

$$
\frac{d}{dt} \log p_t(\mathbf{x}) = \frac{\partial}{\partial t} \log p_t(\mathbf{x}) + \mathbf{u}_t(\mathbf{x}) \cdot \nabla_{\mathbf{x}} \log p_t(\mathbf{x})
$$

where:
- $p_t(\mathbf{x}) = \mathcal{N}(\mathbf{x}; (2t-1)\boldsymbol{\mu}, \sigma_t^2 \mathbf{I})$,
- $\sigma_t^2 = 2t^2 - 2t + 1$,
- $\mathbf{u}_t(\mathbf{x}) = \dfrac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{\sigma_t^2}$,
- $\boldsymbol{\mu} = [10, 0]$, and the dimension $d = 2$.

### Step 1: Compute $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$
The log-density is:
$$
\log p_t(\mathbf{x}) = -\frac{d}{2} \log(2\pi) - \frac{d}{2} \log(\sigma_t^2) - \frac{1}{2\sigma_t^2} \|\mathbf{x} - (2t-1)\boldsymbol{\mu}\|^2
$$
The gradient with respect to $\mathbf{x}$ is:
$$
\nabla_{\mathbf{x}} \log p_t(\mathbf{x}) = -\frac{1}{\sigma_t^2} (\mathbf{x} - (2t-1)\boldsymbol{\mu})
$$

### Step 2: Compute $\frac{\partial}{\partial t} \log p_t(\mathbf{x})$
Differentiate $\log p_t(\mathbf{x})$ with respect to $t$, treating $\mathbf{x}$ as fixed:
$$
\frac{\partial}{\partial t} \log p_t(\mathbf{x}) = -\frac{d}{2} \frac{1}{\sigma_t^2} \frac{\partial \sigma_t^2}{\partial t} - \frac{\partial}{\partial t} \left( \frac{1}{2\sigma_t^2} \|\mathbf{x} - (2t-1)\boldsymbol{\mu}\|^2 \right)
$$
where $\frac{\partial \sigma_t^2}{\partial t} = 4t - 2$. After simplification:
$$
\frac{\partial}{\partial t} \log p_t(\mathbf{x}) = -d \frac{2t-1}{\sigma_t^2} + \frac{2t-1}{\sigma_t^4} \|\mathbf{x} - (2t-1)\boldsymbol{\mu}\|^2 + \frac{2}{\sigma_t^2} (\mathbf{x} - (2t-1)\boldsymbol{\mu}) \cdot \boldsymbol{\mu}
$$

### Step 3: Compute $\mathbf{u}_t(\mathbf{x}) \cdot \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$
Substitute the expressions:
$$
\mathbf{u}_t(\mathbf{x}) \cdot \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) = \left( \frac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{\sigma_t^2} \right) \cdot \left( -\frac{1}{\sigma_t^2} (\mathbf{x} - (2t-1)\boldsymbol{\mu}) \right)
$$
Simplify to:
$$
\mathbf{u}_t(\mathbf{x}) \cdot \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) = -\frac{1}{\sigma_t^4} \left[ (2t-1)\mathbf{x} + \boldsymbol{\mu} \right] \cdot \left[ \mathbf{x} - (2t-1)\boldsymbol{\mu} \right]
$$

### Step 4: Sum the terms
Combine both parts:
$$
\frac{d}{dt} \log p_t(\mathbf{x}) = \left[ -d \frac{2t-1}{\sigma_t^2} + \frac{2t-1}{\sigma_t^4} \|\mathbf{x} - (2t-1)\boldsymbol{\mu}\|^2 + \frac{2}{\sigma_t^2} (\mathbf{x} - (2t-1)\boldsymbol{\mu}) \cdot \boldsymbol{\mu} \right] + \left[ -\frac{1}{\sigma_t^4} \left[ (2t-1)\mathbf{x} + \boldsymbol{\mu} \right] \cdot \left[ \mathbf{x} - (2t-1)\boldsymbol{\mu} \right] \right]
$$
After algebraic simplification (where all $\mathbf{x}$-dependent terms cancel), the result is:
$$
\frac{d}{dt} \log p_t(\mathbf{x}) = -d \frac{2t-1}{\sigma_t^2}
$$

### Final Result
For $d = 2$ and $\sigma_t^2 = 2t^2 - 2t + 1$:
$$
\boxed{\dfrac{d}{dt} \log p_{t}(\mathbf{x}) = -2 \cdot \dfrac{2t - 1}{2t^{2} - 2t + 1}}
$$

### Verification via Continuity Equation
The continuity equation requires:
$$
\frac{d}{dt} \log p_t(\mathbf{x}) = - \nabla \cdot \mathbf{u}_t(\mathbf{x})
$$
Compute the divergence:
$$
\nabla \cdot \mathbf{u}_t(\mathbf{x}) = \nabla \cdot \left( \dfrac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{\sigma_t^2} \right) = \frac{(2t-1)}{\sigma_t^2} \nabla \cdot \mathbf{x} = \frac{(2t-1) \cdot d}{\sigma_t^2}
$$
Thus:
$$
- \nabla \cdot \mathbf{u}_t(\mathbf{x}) = -d \frac{2t-1}{\sigma_t^2}
$$
which matches the result above, confirming correctness. The total derivative is independent of $\mathbf{x}$, a special property of this Gaussian flow.


## Appendix E

The form $\mathbf{u}_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \frac{\dot{\sigma}_t^{2}}{2\sigma_t^{2}} (\mathbf{x} - \boldsymbol{\mu}_t)$ **holds generally for isotropic Gaussians** $p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \sigma_0^2 \mathbf{I})$ and $p_1 = \mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2 \mathbf{I})$ under independent coupling. Here's the derivation and verification:

---

### **Step 1: Marginal Distribution at Time $t$**
Under independent coupling:
- $\mathbf{x}_0 \sim \mathcal{N}(\boldsymbol{\mu}_0, \sigma_0^2 \mathbf{I})$
- $\mathbf{x}_1 \sim \mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2 \mathbf{I})$

With linear interpolation $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$, the marginal is Gaussian:
$$
\mathbf{x}_t \sim \mathcal{N}(\boldsymbol{\mu}_t, \sigma_t^2 \mathbf{I}),
$$
where:
- **Mean**: $\boldsymbol{\mu}_t = (1-t)\boldsymbol{\mu}_0 + t\boldsymbol{\mu}_1$
- **Variance**: $\sigma_t^2 = (1-t)^2 \sigma_0^2 + t^2 \sigma_1^2$

---

### **Step 2: Conditional Expectations**
Given $\mathbf{x}_t = \mathbf{x}$, the posterior expectations are:
$$
\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}] = \boldsymbol{\mu}_0 + \frac{(1-t)\sigma_0^2}{\sigma_t^2} (\mathbf{x} - \boldsymbol{\mu}_t)
$$
$$
\mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}] = \boldsymbol{\mu}_1 + \frac{t\sigma_1^2}{\sigma_t^2} (\mathbf{x} - \boldsymbol{\mu}_t)
$$

---

### **Step 3: Marginal Vector Field**
$$
\mathbf{u}_t(\mathbf{x}) = \mathbb{E}[\mathbf{x}_1 - \mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}] = \left[ \boldsymbol{\mu}_1 + \frac{t\sigma_1^2}{\sigma_t^2} (\mathbf{x} - \boldsymbol{\mu}_t) \right] - \left[ \boldsymbol{\mu}_0 + \frac{(1-t)\sigma_0^2}{\sigma_t^2} (\mathbf{x} - \boldsymbol{\mu}_t) \right]
$$
Simplify:
$$
\mathbf{u}_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \frac{t\sigma_1^2 - (1-t)\sigma_0^2}{\sigma_t^2} (\mathbf{x} - \boldsymbol{\mu}_t)
$$

---

### **Step 4: Relate to $\dot{\sigma}_t^2$**
Compute the derivative of $\sigma_t^2$:
$$
\dot{\sigma}_t^2 = \frac{d}{dt} \left[ (1-t)^2 \sigma_0^2 + t^2 \sigma_1^2 \right] = -2(1-t)\sigma_0^2 + 2t\sigma_1^2 = 2 \left[ t\sigma_1^2 - (1-t)\sigma_0^2 \right]
$$
Thus:
$$
t\sigma_1^2 - (1-t)\sigma_0^2 = \frac{\dot{\sigma}_t^2}{2}
$$
Substitute into $\mathbf{u}_t(\mathbf{x})$:
$$
\boxed{\mathbf{u}_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \dfrac{\dot{\sigma}_t^{2}}{2\sigma_t^{2}} (\mathbf{x} - \boldsymbol{\mu}_t)}
$$

---

### **Verification at Boundaries**
1. **At $t = 0$**:
   - $\sigma_t^2 = \sigma_0^2$, $\dot{\sigma}_t^2 = -2\sigma_0^2$, $\boldsymbol{\mu}_t = \boldsymbol{\mu}_0$
   - $\mathbf{u}_0(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \frac{-2\sigma_0^2}{2\sigma_0^2} (\mathbf{x} - \boldsymbol{\mu}_0) = \boldsymbol{\mu}_1 - \mathbf{x}$
   - **Matches** $\mathbb{E}[\mathbf{x}_1 - \mathbf{x}_0 \mid \mathbf{x}_0 = \mathbf{x}] = \boldsymbol{\mu}_1 - \mathbf{x}$.

1. **At $t = 1$**:
   - $\sigma_t^2 = \sigma_1^2$, $\dot{\sigma}_t^2 = 2\sigma_1^2$, $\boldsymbol{\mu}_t = \boldsymbol{\mu}_1$
   - $\mathbf{u}_1(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \frac{2\sigma_1^2}{2\sigma_1^2} (\mathbf{x} - \boldsymbol{\mu}_1) = \mathbf{x} - \boldsymbol{\mu}_0$
   - **Matches** $\mathbb{E}[\mathbf{x}_1 - \mathbf{x}_0 \mid \mathbf{x}_1 = \mathbf{x}] = \mathbf{x} - \boldsymbol{\mu}_0$.

---

### **Key Observations**
- **Generalization**: The form holds for arbitrary $\sigma_0^2, \sigma_1^2 > 0$, reducing to the unit-variance case when $\sigma_0^2 = \sigma_1^2 = 1$.
- **Role of $\dot{\sigma}_t^2$**: The term $\frac{\dot{\sigma}_t^2}{2\sigma_t^2}$ captures the **time-dependent scaling** of the drift relative to the current dispersion $\sigma_t^2$.
- **Interpretation**: The vector field transports mass from $p_0$ to $p_1$ by:
  - A constant velocity $(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)$ (mean shift),
  - A position-dependent correction that contracts/expands dispersion based on $\dot{\sigma}_t^2$.

This result is consistent with probability flow ODEs in diffusion models and holds for any isotropic Gaussians under independent coupling.

## Appendix F

为了求解常微分方程（ODE）：
$$
\frac{d\mathbf{x}}{dt} = \mathbf{u}_t(\mathbf{x})
$$
其中向量场 $\mathbf{u}_t(\mathbf{x})$ 在独立耦合下定义为：
$$
\mathbf{u}_t(\mathbf{x}) = \dot{\boldsymbol{\mu}}_t  + \frac{\dot{\sigma}_t^{2}}{2\sigma_t^{2}} (\mathbf{x} - \boldsymbol{\mu}_t)
$$
且初始条件为 $\mathbf{x}(0) = \mathbf{x}_0$。这里，$\boldsymbol{\mu}_t = (1-t)\boldsymbol{\mu}_0 + t\boldsymbol{\mu}_1$，$\sigma_t^2 = (1-t)^2 \sigma_0^2 + t^2 \sigma_1^2$，$\dot{\sigma}_t^2 = \frac{d}{dt}\sigma_t^2 = -2(1-t)\sigma_0^2 + 2t\sigma_1^2$。

### 推导过程
1. **变量变换**：  
   令 $\mathbf{y} = \mathbf{x} - \boldsymbol{\mu}_t$。则：
   $$
   \frac{d\mathbf{y}}{dt} = \frac{d\mathbf{x}}{dt} - \frac{d\boldsymbol{\mu}_t}{dt}
   $$
   计算 $\frac{d\boldsymbol{\mu}_t}{dt}$：
   $$
   \frac{d\boldsymbol{\mu}_t}{dt} = -\boldsymbol{\mu}_0 + \boldsymbol{\mu}_1 = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0
   $$
   代入 ODE：
   $$
   \frac{d\mathbf{y}}{dt} = \left[ (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \frac{\dot{\sigma}_t^{2}}{2\sigma_t^{2}} \mathbf{y} \right] - (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) = \frac{\dot{\sigma}_t^{2}}{2\sigma_t^{2}} \mathbf{y}
   $$
   得到简化方程：
   $$
   \frac{d\mathbf{y}}{dt} = \frac{\dot{\sigma}_t^{2}}{2\sigma_t^{2}} \mathbf{y}
   $$

2. **求解简化 ODE**：  
   该方程为可分离变量：
   $$
   \frac{d\mathbf{y}}{\mathbf{y}} = \frac{\dot{\sigma}_t^{2}}{2\sigma_t^{2}}  dt
   $$
   其中 $\frac{\dot{\sigma}_t^{2}}{2\sigma_t^{2}} = \frac{1}{2} \frac{d}{dt} \ln \sigma_t^2$。积分两边：
   $$
   \int \frac{d\mathbf{y}}{\mathbf{y}} = \frac{1}{2} \int d(\ln \sigma_t^2)
   $$
   得：
   $$
   \ln |\mathbf{y}| = \frac{1}{2} \ln \sigma_t^2 + C
   $$
   其中 $C$ 为积分常数。解出 $\mathbf{y}$:
   $$
   \mathbf{y}(t) = \mathbf{y}(0) \exp\left( \frac{1}{2} \ln \frac{\sigma_t^2}{\sigma_0^2} \right) = \mathbf{y}(0) \left( \frac{\sigma_t^2}{\sigma_0^2} \right)^{1/2} = \mathbf{y}(0) \frac{\sigma_t}{\sigma_0}
   $$
   这里 $\sigma_t = \sqrt{\sigma_t^2}$，$\sigma_0 = \sqrt{\sigma_0^2}$ 为标准差。

3. **初始条件代入**：  
   在 $t = 0$ 时，$\mathbf{y}(0) = \mathbf{x}(0) - \boldsymbol{\mu}_t(0) = \mathbf{x}_0 - \boldsymbol{\mu}_0$。因此：
   $$
   \mathbf{y}(t) = (\mathbf{x}_0 - \boldsymbol{\mu}_0) \frac{\sigma_t}{\sigma_0}
   $$

4. **还原变量**：  
   由 $\mathbf{y}(t) = \mathbf{x}(t) - \boldsymbol{\mu}_t$，得：
   $$
   \mathbf{x}(t) = \boldsymbol{\mu}_t + (\mathbf{x}_0 - \boldsymbol{\mu}_0) \frac{\sigma_t}{\sigma_0}
   $$
   代入 $\boldsymbol{\mu}_t = (1-t)\boldsymbol{\mu}_0 + t\boldsymbol{\mu}_1$ 和 $\sigma_t = \sqrt{(1-t)^2 \sigma_0^2 + t^2 \sigma_1^2}$，最终解为：
   $$
   \mathbf{x}(t) = \left[(1-t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1\right] + \frac{\sqrt{(1-t)^2 \sigma_0^2 + t^2 \sigma_1^2}}{\sigma_0} (\mathbf{x}_0 - \boldsymbol{\mu}_0)
   $$

### 验证
- **初始条件 $t = 0$**：  
  $\boldsymbol{\mu}_t = \boldsymbol{\mu}_0$，$\sigma_t = \sigma_0$，  
  $\mathbf{x}(0) = \boldsymbol{\mu}_0 + \frac{\sigma_0}{\sigma_0} (\mathbf{x}_0 - \boldsymbol{\mu}_0) = \mathbf{x}_0$，满足初始条件。

- **分布验证**：  
  若 $\mathbf{x}_0 \sim \mathcal{N}(\boldsymbol{\mu}_0, \sigma_0^2 \mathbf{I})$，则 $\mathbf{x}(t) \sim \mathcal{N}(\boldsymbol{\mu}_t, \sigma_t^2 \mathbf{I})$，符合流匹配的边际分布要求。

### 最终解
$$
\boxed{\mathbf{x}(t) = \left[(1-t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1\right] + \dfrac{\sqrt{(1-t)^2 \sigma_0^2 + t^2 \sigma_1^2}}{\sigma_0} (\mathbf{x}_0 - \boldsymbol{\mu}_0)}
$$

## Appendix G

To compute the marginal vector field $\mathbf{u}_t(\mathbf{x})$ for flow matching between two Gaussians $p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \Sigma_0)$ and $p_1 = \mathcal{N}(\boldsymbol{\mu}_1, \Sigma_1)$ under **independent coupling** (cross-covariance = 0), we start from the conditional vector field $\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1)$ and derive the marginal field through expectation. The conditional field is derived from the straight-line path $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$:

### Step 1: Conditional Vector Field
The time derivative of the path gives the conditional vector field:
$$
\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) = \frac{d\mathbf{x}_t}{dt} = \mathbf{x}_1 - \mathbf{x}_0
$$
Expressing $\mathbf{x}_0$ in terms of $\mathbf{x}_t$ and $\mathbf{x}_1$:
$$
\mathbf{x}_0 = \frac{\mathbf{x}_t - t\mathbf{x}_1}{1-t}
$$
Substitute to eliminate $\mathbf{x}_0$:
$$
\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) = \mathbf{x}_1 - \frac{\mathbf{x} - t\mathbf{x}_1}{1-t} = \frac{\mathbf{x}_1 - \mathbf{x}}{1-t}
$$

### Step 2: Marginal Vector Field
The marginal vector field is the expectation over $\mathbf{x}_1$ conditioned on $\mathbf{x}$:
$$
\mathbf{u}_t(\mathbf{x}) = \mathbb{E}_{p_t(\mathbf{x}_1 \mid \mathbf{x})} \left[ \mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) \right] = \mathbb{E}_{p_t(\mathbf{x}_1 \mid \mathbf{x})} \left[ \frac{\mathbf{x}_1 - \mathbf{x}}{1-t} \right] = \frac{1}{1-t} \left( \mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}] - \mathbf{x} \right)
$$

### Step 3: Compute $\mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}]$
Under independent coupling, the joint distribution of $\mathbf{x}_t$ and $\mathbf{x}_1$ is Gaussian:
$$
\begin{pmatrix} \mathbf{x}_t \\ \mathbf{x}_1 \end{pmatrix} \sim \mathcal{N} \left( \begin{pmatrix} \boldsymbol{\mu}_t \\ \boldsymbol{\mu}_1 \end{pmatrix}, \begin{pmatrix} \Sigma_t & \text{Cov}(\mathbf{x}_t, \mathbf{x}_1) \\ \text{Cov}(\mathbf{x}_1, \mathbf{x}_t) & \Sigma_1 \end{pmatrix} \right)
$$
where:
- $\boldsymbol{\mu}_t = (1-t)\boldsymbol{\mu}_0 + t\boldsymbol{\mu}_1$
- $\Sigma_t = (1-t)^2 \Sigma_0 + t^2 \Sigma_1$ (independent coupling)
- $\text{Cov}(\mathbf{x}_t, \mathbf{x}_1) = t\Sigma_1$ (since $\mathbf{x}_0$ and $\mathbf{x}_1$ are independent)

The conditional expectation is:
$$
\mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}] = \boldsymbol{\mu}_1 + \text{Cov}(\mathbf{x}_1, \mathbf{x}_t) \Sigma_t^{-1} (\mathbf{x} - \boldsymbol{\mu}_t) = \boldsymbol{\mu}_1 + t\Sigma_1 \Sigma_t^{-1} (\mathbf{x} - \boldsymbol{\mu}_t)
$$

### Step 4: Substitute into $\mathbf{u}_t(\mathbf{x})$
$$
\mathbf{u}_t(\mathbf{x}) = \frac{1}{1-t} \left( \boldsymbol{\mu}_1 + t\Sigma_1 \Sigma_t^{-1} (\mathbf{x} - \boldsymbol{\mu}_t) - \mathbf{x} \right)
$$
Rewrite $\boldsymbol{\mu}_1 - \mathbf{x}$ as:
$$
\boldsymbol{\mu}_1 - \mathbf{x} = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_t) + (\boldsymbol{\mu}_t - \mathbf{x}) = (1-t)(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + (\boldsymbol{\mu}_t - \mathbf{x})
$$
Substitute:
$$
\mathbf{u}_t(\mathbf{x}) = \frac{1}{1-t} \left( (1-t)(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + (\boldsymbol{\mu}_t - \mathbf{x}) + t\Sigma_1 \Sigma_t^{-1} (\mathbf{x} - \boldsymbol{\mu}_t) \right)
$$
Simplify:
$$
\mathbf{u}_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \frac{1}{1-t} \left( -(\mathbf{x} - \boldsymbol{\mu}_t) + t\Sigma_1 \Sigma_t^{-1} (\mathbf{x} - \boldsymbol{\mu}_t) \right)
$$
Factor:
$$
\mathbf{u}_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \frac{ -I + t\Sigma_1 \Sigma_t^{-1} }{1-t} (\mathbf{x} - \boldsymbol{\mu}_t)
$$

### Step 5: Verify Consistency
Using the continuity equation for the Gaussian path $p_t = \mathcal{N}(\boldsymbol{\mu}_t, \Sigma_t)$:
$$
\mathbf{u}_t(\mathbf{x}) = \dot{\boldsymbol{\mu}}_t + \frac{1}{2} \dot{\Sigma}_t \Sigma_t^{-1} (\mathbf{x} - \boldsymbol{\mu}_t)
$$
where:
- $\dot{\boldsymbol{\mu}}_t = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$
- $\dot{\Sigma}_t = -2(1-t)\Sigma_0 + 2t\Sigma_1$

Substitute:
$$
\mathbf{u}_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \frac{1}{2} \left( -2(1-t)\Sigma_0 + 2t\Sigma_1 \right) \Sigma_t^{-1} (\mathbf{x} - \boldsymbol{\mu}_t) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \left( t\Sigma_1 - (1-t)\Sigma_0 \right) \Sigma_t^{-1} (\mathbf{x} - \boldsymbol{\mu}_t)
$$
This matches the expression from Step 4 since:
$$
t\Sigma_1 - (1-t)\Sigma_0 = \frac{ -I + t\Sigma_1 \Sigma_t^{-1} }{1-t} \cdot \Sigma_t
$$

### Final Result
The marginal vector field for independent coupling is:
$$
\boxed{\mathbf{u}_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \dfrac{ t \Sigma_1 - (1-t) \Sigma_0 }{ \Sigma_t } (\mathbf{x} - \boldsymbol{\mu}_t)}
$$
where $\Sigma_t = (1-t)^2 \Sigma_0 + t^2 \Sigma_1$.

### Special Case: Isotropic Gaussians
If $\Sigma_0 = \sigma_0^2 \mathbf{I}$ and $\Sigma_1 = \sigma_1^2 \mathbf{I}$:
$$
\Sigma_t = \sigma_t^2 \mathbf{I}, \quad \sigma_t^2 = (1-t)^2 \sigma_0^2 + t^2 \sigma_1^2
$$
$$
\dot{\sigma}_t^2 = -2(1-t)\sigma_0^2 + 2t\sigma_1^2
$$
The vector field simplifies to:
$$
\mathbf{u}_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \frac{\dot{\sigma}_t^2}{2\sigma_t^2} (\mathbf{x} - \boldsymbol{\mu}_t)
$$

## Appendix H:  General Gaussian Flow

To determine the trajectory $\mathbf{x}(t)$ for the flow defined by the vector field $\mathbf{u}_t(\mathbf{x})$ under independent coupling (cross-covariance = 0) between Gaussians $p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \Sigma_0)$ and $p_1 = \mathcal{N}(\boldsymbol{\mu}_1, \Sigma_1)$, we start from the given vector field and solve the associated ordinary differential equation (ODE). The vector field is:

$$
\mathbf{u}_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \left( t \Sigma_1 - (1-t) \Sigma_0 \right) \Sigma_t^{-1} (\mathbf{x} - \boldsymbol{\mu}_t),
$$

where:
- $\boldsymbol{\mu}_t = (1-t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1$,
- $\Sigma_t = (1-t)^2 \Sigma_0 + t^2 \Sigma_1$.

The trajectory $\mathbf{x}(t)$ satisfies the ODE:
$$
\frac{d\mathbf{x}}{dt} = \mathbf{u}_t(\mathbf{x}), \quad \mathbf{x}(0) = \mathbf{x}_0.
$$

This is a linear, non-autonomous ODE. To solve it, we decompose $\mathbf{x}(t)$ into its mean and deviation components. Define:
$$
\mathbf{y}(t) = \mathbf{x}(t) - \boldsymbol{\mu}_t,
$$
where $\mathbf{y}(t)$ represents the deviation from the time-dependent mean $\boldsymbol{\mu}_t$. The initial condition is $\mathbf{y}(0) = \mathbf{x}_0 - \boldsymbol{\mu}_0$.

### Step 1: Derive the ODE for $\mathbf{y}(t)$
Differentiate $\mathbf{y}(t)$:
$$
\frac{d\mathbf{y}}{dt} = \frac{d\mathbf{x}}{dt} - \dot{\boldsymbol{\mu}}_t,
$$
where $\dot{\boldsymbol{\mu}}_t = \frac{d\boldsymbol{\mu}_t}{dt} = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$. Substitute the ODE for $\frac{d\mathbf{x}}{dt}$:
$$
\frac{d\mathbf{y}}{dt} = \mathbf{u}_t(\mathbf{x}) - \dot{\boldsymbol{\mu}}_t = \left[ (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \left( t \Sigma_1 - (1-t) \Sigma_0 \right) \Sigma_t^{-1} \mathbf{y} \right] - (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) = \left( t \Sigma_1 - (1-t) \Sigma_0 \right) \Sigma_t^{-1} \mathbf{y}.
$$

Simplify the coefficient:
$$
\mathbf{B}_t = \left( t \Sigma_1 - (1-t) \Sigma_0 \right) \Sigma_t^{-1}.
$$
Thus, the ODE for $\mathbf{y}(t)$ is:
$$
\frac{d\mathbf{y}}{dt} = \mathbf{B}_t \mathbf{y}, \quad \mathbf{y}(0) = \mathbf{x}_0 - \boldsymbol{\mu}_0.
$$

### Step 2: Solve the ODE for $\mathbf{y}(t)$
The solution to $\frac{d\mathbf{y}}{dt} = \mathbf{B}_t \mathbf{y}$ is:
$$
\mathbf{y}(t) = \mathbf{C}(t) \mathbf{y}(0),
$$
where $\mathbf{C}(t)$ is the fundamental matrix satisfying:
$$
\frac{d\mathbf{C}}{dt} = \mathbf{B}_t \mathbf{C}, \quad \mathbf{C}(0) = \mathbf{I}.
$$

### Step 3: Express $\mathbf{x}(t)$
Substitute back to $\mathbf{x}(t)$:
$$
\mathbf{x}(t) = \boldsymbol{\mu}_t + \mathbf{y}(t) = \boldsymbol{\mu}_t + \mathbf{C}(t) (\mathbf{x}_0 - \boldsymbol{\mu}_0).
$$

### Step 4: Closed-form solution under commutativity (if applicable)
If $\Sigma_0$ and $\Sigma_1$ commute (i.e., $\Sigma_0 \Sigma_1 = \Sigma_1 \Sigma_0$), then $\mathbf{C}(t)$ simplifies to:
$$
\mathbf{C}(t) = \Sigma_t^{1/2} \Sigma_0^{-1/2},
$$
and the solution becomes:
$$
\mathbf{x}(t) = \boldsymbol{\mu}_t + \Sigma_t^{1/2} \Sigma_0^{-1/2} (\mathbf{x}_0 - \boldsymbol{\mu}_0).
$$
This holds because $\frac{d}{dt}(\Sigma_t^{1/2}) = \frac{1}{2} \dot{\Sigma}_t \Sigma_t^{-1/2}$ when $\Sigma_0$ and $\Sigma_1$ commute, satisfying the ODE for $\mathbf{C}(t)$.

### Final Result
The trajectory $\mathbf{x}(t)$ is:
$$
\boxed{\mathbf{x}(t) = \boldsymbol{\mu}_t + \mathbf{C}(t) \left( \mathbf{x}_0 - \boldsymbol{\mu}_0 \right)}
$$
where:
- $\boldsymbol{\mu}_t = (1-t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1$,
- $\Sigma_t = (1-t)^2 \Sigma_0 + t^2 \Sigma_1$,
- $\mathbf{C}(t)$ solves the matrix ODE:
  $$
  \frac{d\mathbf{C}}{dt} = \left( t \Sigma_1 - (1-t) \Sigma_0 \right) \Sigma_t^{-1} \mathbf{C}, \quad \mathbf{C}(0) = \mathbf{I}.
  $$

If $\Sigma_0$ and $\Sigma_1$ commute, this simplifies to:
$$
\boxed{\mathbf{x}(t) = \boldsymbol{\mu}_t + \Sigma_t^{1/2} \Sigma_0^{-1/2} \left( \mathbf{x}_0 - \boldsymbol{\mu}_0 \right)}
$$

### Key Notes:
- **General Case**: The ODE for $\mathbf{C}(t)$ must be solved numerically if $\Sigma_0$ and $\Sigma_1$ do not commute.
- **Behavior**: The solution ensures that the marginal distribution of $\mathbf{x}(t)$ is $\mathcal{N}(\boldsymbol{\mu}_t, \Sigma_t)$ under independent coupling.
- **Initial Condition**: At $t=0$, $\mathbf{x}(0) = \boldsymbol{\mu}_0 + \mathbf{I}(\mathbf{x}_0 - \boldsymbol{\mu}_0) = \mathbf{x}_0$.
- **Endpoint**: At $t=1$, $\mathbf{x}(1) = \boldsymbol{\mu}_1 + \mathbf{C}(1) (\mathbf{x}_0 - \boldsymbol{\mu}_0)$, where $\mathbf{C}(1)$ depends on the solution to the ODE.