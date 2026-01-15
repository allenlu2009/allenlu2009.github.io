---
title: Math AI - Expand Score Matching to Flow Matching
date: 2025-06-15 23:10:08
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





## Flow Sampling (from $\mathbf{x}_0$ and $u_t$ to get $\mathbf{x}_t$)

**最重要的是 $\mathbf{x}_t$不是直線**！因為 $\mathbf{u}_t(\mathbf{x}_t)$ 是一個平均的結果，不是一個 constant vector!
但是 condition vector 是直線 (in the OT case).   

理論上非常簡單，就是解一個 ODE with initial condition $\mathbf{x}_0$ is:
$$\frac{d\mathbf{x}}{dt} = \mathbf{u}_t(\mathbf{x})$$ The exact solution for 例一 of the ODE 
$$
\mathbf{x}_t = (2t-1)\boldsymbol{\mu} + \sqrt{2t^2 - 2t + 1} \cdot (\mathbf{x}_0 + \boldsymbol{\mu})
$$
通用的表示：（Two indepedent Gaussians, Appendix E and F)
$$\boxed{\mathbf{u}_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \frac{ {d\log\sigma}_t^{2}}{2 dt} (\mathbf{x} - \boldsymbol{\mu}_t)}$$
$$\begin{aligned}
\mathbf{x}(t) &= \left[(1-t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1\right] + \dfrac{\sqrt{(1-t)^2 \sigma_0^2 + t^2 \sigma_1^2}}{\sigma_0} (\mathbf{x}_0 - \boldsymbol{\mu}_0)\\
\phi_t(\mathbf{x}_0) &= \mathbf{x}(t)=\boldsymbol{\mu}_t + \dfrac{\sigma_t}{\sigma_0} (\mathbf{x}_0 - \boldsymbol{\mu}_0)\\
\end{aligned}
$$
- $\boldsymbol{\mu}_t = (1-t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1 = \boldsymbol{\mu}_0 + t(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)$ 是以 $\boldsymbol{\mu}_0$ 為起點，斜率為 $(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)$ 的直綫。只有在$\mathbf{x}_0 = \boldsymbol{\mu}_0$ 才會走這條直綫。當 $\mathbf{x}_0 \ne \boldsymbol{\mu}_0$ 偏離的部分就會照 standard deviation 比例
 ($\frac{\sigma_t}{\sigma_0}$) 加到這條直綫。 
- $t =0, \phi_0(\mathbf{x}_0) = \mathbf{x}_0$
- $t =1, \phi_1(\mathbf{x}_0) = \boldsymbol{\mu}_1 + \frac{\sigma_1}{\sigma_0}(\mathbf{x}_0-\boldsymbol{\mu}_0)$.   如果 $\sigma_1=\sigma_0=\sigma$ , $\phi_1(\mathbf{x}_0) = \mathbf{x}_0+(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)$.  即是所有的終點都是起點加上 mean difference.

$\boldsymbol{\mu}_t = (1 - t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1$ and $\boldsymbol{\Sigma}_t = (1 - t)^2 \boldsymbol{\Sigma}_0 + t^2 \boldsymbol{\Sigma}_1$
If $\boldsymbol{\Sigma}_0$ and $\boldsymbol{\Sigma}_1$ **commute**, the flow simplifies to: (Appendix H)
$$
\boxed{\phi_t(\mathbf{x}_0)=\mathbf{x}(t) = \boldsymbol{\mu}_t + \boldsymbol{\Sigma}_t^{1/2} \boldsymbol{\Sigma}_0^{-1/2} \left( \mathbf{x}_0 - \boldsymbol{\mu}_0 \right)}
$$

## Mean Flow 的 Idea 就是 Sampling 的時候走直綫！ 這樣只需要 1-step!  初中幾何的概念！

https://www.youtube.com/watch?v=swKdn-qT47Q&ab_channel=%E9%98%BF%E5%86%89

Typical flow model training

![[Pasted image 20250615091041.png]]


Difference between flow matching vs. mean flow matching: one-step generation
![[Pasted image 20250615091217.png]]

Mean flow model training

![[Pasted image 20250615081535.png]]

![[Pasted image 20250615081608.png]]

![[Pasted image 20250615081653.png]]


## How about variable time step mean flow matching?  on manifold, slow, off manifold - fast?



## Gaussian Flow Mean Flow Field

### Correct Closed-Form Vector Fields

**Given:**
- $p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \sigma_0^2 I)$
- $p_1 = \mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2 I)$
- Given $\mathbf{x}_1$, conditional straight-line trajectory: $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t \mathbf{x}_1$
- Marginal distribution: $\mathbf{x}_t \sim \mathcal{N}(\boldsymbol{\mu}_t, \sigma_t^2 I)$  
  where $\boldsymbol{\mu}_t = (1-t)\boldsymbol{\mu}_0 + t\boldsymbol{\mu}_1$  
  and $\sigma_t^2 = (1-t)^2\sigma_0^2 + t^2\sigma_1^2$


Example 1:  Two Gaussians with opposite mean and same variance. (Appendix A)

先看一般 flow matching, 也就是 instantaneous vector field。

$$\boxed{v_t(x) = \dot{\boldsymbol{\mu}}_t  + \dfrac{\dot{\sigma_t^2}}{2\sigma_t^2}(\mathbf{x} - \boldsymbol{\mu}_t)=\boldsymbol{\mu}_1-\boldsymbol{\mu}_0  + \dfrac{-(1-t)\sigma_0^2 + t\sigma_1^2}{(1-t)^2\sigma_0^2 + t^2\sigma_1^2}(\mathbf{x} - \boldsymbol{\mu}_t)} $$
假設 $\mu_1=-\mu$ and $\mu_2=+\mu$,  $\sigma_0=\sigma_1=\sigma$
$$v_t(\mathbf{x}) = 2\boldsymbol{\mu}  + \dfrac{2t-1}{2t^2-2t+1}(\mathbf{x} - (2t-1)\boldsymbol{\mu}) $$
$$v_t(\mathbf{x}) = 2\boldsymbol{\mu}  + \dfrac{\dot{\sigma^2_t}}{2\sigma^2_t}(\mathbf{x} - \boldsymbol{\mu}_t) $$
平均向量場是 $2\boldsymbol{\mu}$，在 $t<0.5$,  $\sigma_t$ 變小，所以微分是負值。大於 mean $\boldsymbol{\mu}_t$ 會被壓縮，小於 mean 則被膨脹。


實際 unconditional trajectory (given $x_0$ and $v_t(x)$): 
兩個重點
- 不是直線，而是曲線
- $t = 0,  v_t = 2 \mu - (x_0 + \mu) = \mu-x_0$,  其實是 $E(x_1) - x_0$ 
- $t = 0.5,  v_t = 2 \mu$,  其實是 $E(x_1) - E(x_0)$ 
- $t = 1,  v_t = 2 \mu + (x_1 - \mu) = x_1-\mu$,  
- 
- $t=1$,  $\mathbf{x}_1 = \mathbf{x}_0 + 2\boldsymbol{\mu}$,  也就是雖然是曲線，最後的終端相當於平均平移 $2\boldsymbol{\mu}$ 
$$
\boxed{ \mathbf{x}_t = \boldsymbol{\mu}_t + \frac{ {\sigma}_t} { {\sigma}_0} \left( \mathbf{x}_0 - \boldsymbol{\mu}_0 \right)=(2t-1) \boldsymbol{\mu}+\sqrt{2t^2-2t+1}\left( \mathbf{x}_0 + \boldsymbol{\mu} \right)}
$$



$$
\boxed{u(z_t, t_{\text{end}}, t) = 2\boldsymbol{\mu} + \dfrac{\sqrt{2t_{\text{end}}^2 - 2t_{\text{end}} + 1} - \sqrt{2t^2 - 2t + 1}}{\sqrt{2t^2 - 2t + 1} \cdot (t_{\text{end}} - t)} \left( z_t - (2t - 1) \boldsymbol{\mu} \right)}
$$
Assume $t_{end}=1$

$u(z_t, t_{\text{end}}, t) = 2\boldsymbol{\mu} + \dfrac{1 - \sqrt{2t^2 - 2t + 1}}{\sqrt{2t^2 - 2t + 1} \cdot (1 - t)} \left( z_t - (2t - 1) \boldsymbol{\mu} \right)=2\boldsymbol{\mu} + \dfrac{1 - \sqrt{2t^2 - 2t + 1}}{ {2t^2 - 2t + 1} \cdot (1 - t)} \left( z_t - (2t - 1) \boldsymbol{\mu} \right)$

所以 $u(\mathbf{x}_0, 1, 0) = 2\boldsymbol{\mu}$,  也就是 $\mathbf{x}_1= \mathbf{x}_0+2\boldsymbol{\mu}$, 和之前一樣。



---

### 1. Instantaneous Flow Matching Vector Field


**Given:**
- $p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \sigma_0^2 I)$
- $p_1 = \mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2 I)$
- Marginal distribution: $\mathbf{x}_t \sim \mathcal{N}(\boldsymbol{\mu}_t, \sigma_t^2 I)$  
  where $\boldsymbol{\mu}_t = (1-t)\boldsymbol{\mu}_0 + t\boldsymbol{\mu}_1$  
  and $\sigma_t^2 = (1-t)^2\sigma_0^2 + t^2\sigma_1^2$


Example 1:  Two Gaussians with opposite mean and same variance. (Appendix A)

先看一般 flow matching, 也就是 instantaneous vector field。

$$\boxed{\mathbf{v}_t(\mathbf{x}) = \dot{\boldsymbol{\mu}}_t  + \dfrac{\dot{\sigma_t^2}}{2\sigma_t^2}(\mathbf{x} - \boldsymbol{\mu}_t)=\dot{\boldsymbol{\mu}}_t  + \dfrac{\dot{\sigma_t}}{\sigma_t}(\mathbf{x} - \boldsymbol{\mu}_t)=\boldsymbol{\mu}_1-\boldsymbol{\mu}_0  + \dfrac{-(1-t)\sigma_0^2 + t\sigma_1^2}{(1-t)^2\sigma_0^2 + t^2\sigma_1^2}(\mathbf{x} - \boldsymbol{\mu}_t)} $$

對應的 trajectory 是:
$$
\boxed{ \mathbf{x}_t = \boldsymbol{\mu}_t + \frac{ {\sigma}_t} { {\sigma}_0} \left( \mathbf{x}_0 - \boldsymbol{\mu}_0 \right)}
$$
prove $\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_t(\mathbf{x})$

$$\frac{\mathbf{x}_t - \boldsymbol{\mu}_t}{ {\sigma}_t} = \frac{\mathbf{x}_0 - \boldsymbol{\mu}_0} { {\sigma}_0} $$
### Core Intuition: **Constant Normalized Offset or z-score**

- **Left side**: $x_t = \mu_t + \frac{\sigma_t}{\sigma_0}(x_0 - \mu_0)$​​ is the **z-score** of xtxt​ at time tt.  
    It measures how many standard deviations (σtσt​) the point xtxt​ is from the current mean μtμt​.
    
- **Right side**: x0−μ0σ0σ0​x0​−μ0​​ is the **z-score** of x0x0​ at time t=0t=0.
    

The equality states that:  
**The z-score (normalized position relative to the current distribution) remains constant along the entire trajectory**.


The **correct general form** for isotropic Gaussians is:
$$\boxed{v_t(x) = \underbrace{\dot{\boldsymbol{\mu}}_t}_{\text{mean velocity}} + \underbrace{\frac{\dot\sigma_t}{\sigma_t}}_{\text{radial scale}} (x - \boldsymbol{\mu}_t)}$$

where we compute the derivatives:
1. $\dot{\boldsymbol{\mu}}_t = \frac{d}{dt}\boldsymbol{\mu}_t = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$
2. $\dot\sigma_t^2 = \frac{d}{dt}\sigma_t^2 = 2(1-t)\sigma_0^2(-1) + 2t\sigma_1^2 = -2(1-t)\sigma_0^2 + 2t\sigma_1^2$  
   → $\dot\sigma_t = \frac{\dot\sigma_t^2}{2\sigma_t} = \dfrac{-(1-t)\sigma_0^2 + t\sigma_1^2}{\sigma_t}$

**Final form:**
$$\boxed{v_t(x) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \left(\dfrac{-(1-t)\sigma_0^2 + t\sigma_1^2}{(1-t)^2\sigma_0^2 + t^2\sigma_1^2}\right)(x - \boldsymbol{\mu}_t)}$$


$$\boxed{\mathbf{v}_t(\mathbf{x}_t) = \dot{\boldsymbol{\mu}}_t  + \dfrac{\dot{\sigma_t^2}}{2\sigma_t^2}(\mathbf{x}_t - \boldsymbol{\mu}_t)=\dot{\boldsymbol{\mu}}_t  + \dfrac{\dot{\sigma_t}}{\sigma_t}(\mathbf{x}_t - \boldsymbol{\mu}_t)} $$

對應的 trajectory 是:
$$
\boxed{ \mathbf{x}_t = \boldsymbol{\mu}_t + \frac{ {\sigma}_t} { {\sigma}_0} \left( \mathbf{x}_0 - \boldsymbol{\mu}_0 \right)}
$$


---

### 2. Mean Flow Matching Vector Field (Average Velocity)
For an interval $[t, t_{\text{end}}]$, the average velocity is:
$$
u(t, t_{\text{end}}, x) = \frac{1}{t_{\text{end}} - t} \int_t^{t_{\text{end}}} v_\tau(x)  d\tau
$$

**Key simplification:**  
Because the vector field $v_t(x)$ is **linear** in $x$ (affine transformation), the average velocity retains the same functional form:
$$\boxed{u(t, t_{\text{end}}, x) = v_t(x)}$$

---

### Why Are They Identical?
1. **Path linearity**:  
   $\mathbf{x}_t$ evolves linearly → instantaneous velocity $v_t(x)$ is constant along each path.
   $$
   \frac{d\mathbf{x}_t}{dt} = \mathbf{x}_1 - \mathbf{x}_0 \quad \text{(constant for fixed } \mathbf{x}_0, \mathbf{x}_1)
   $$
2. **Averaging constant velocity**:  
   For any interval $[t, t_{\text{end}}]$:
   $$
   u = \frac{1}{t_{\text{end}}-t} \int_t^{t_{end}} (\mathbf{x}_1 - \mathbf{x}_0)  d\tau = \mathbf{x}_1- \mathbf{x}_0
   $$
3. **Conditional expectation**:  
   Both fields reduce to:
   $$
   \mathbb{E}[\mathbf{x}_1 - \mathbf{x}_0 \mid \mathbf{x}_t] = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \left(\dfrac{-(1-t)\sigma_0^2 + t\sigma_1^2}{(1-t)^2\sigma_0^2 + t^2\sigma_1^2}\right)(\mathbf{x}_t - \boldsymbol{\mu}_t)
   $$

---

### Special Case Verification
**Variance-preserving flow** ($\sigma_0 = \sigma_1 = \sigma$):
- $\dot\sigma_t = 0$  
- $v_t(x) = u(t, t_{\text{end}}, x) = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$  
- Matches expectation: $\mathbb{E}[\mathbf{x}_1 - \mathbf{x}_0] = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$

**Data-to-noise** ($\sigma_0 = \sigma > 0, \sigma_1 = 0$):
- $\boldsymbol{\mu}_t = (1-t)\boldsymbol{\mu}_0$  
- $\sigma_t^2 = (1-t)^2\sigma^2$  
- $\dot\sigma_t = -\sigma(1-t)$  
- $v_t(x) = u(t, t_{\text{end}}, x) = \dfrac{\boldsymbol{\mu}_0 - x}{1-t}$  
- Matches the OU process limit.

---

### Summary
| Property            | Flow Matching                                                                                | Mean Flow Matching                 |
| ------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------- |
| **Form**            | $v_t(x) = \dot{\boldsymbol{\mu}}_t + \dfrac{\dot\sigma_t}{\sigma_t}(x - \boldsymbol{\mu}_t)$ | Identical to $v_t(x)$              |
| **Time Dependence** | Explicit $t$ dependence                                                                      | Independent of $t_{\text{end}}$    |
| **Sampling**        | Requires ODE integration                                                                     | One-step: $\mathbf{x}_1 = \mathbf{x}_0 + u(0,1,\mathbf{x}_0)$ |
| **Complexity**      | Same closed form                                                                             | Same closed form                   |

**Key Insight**:  
For Gaussian distributions with linear interpolation paths, the average velocity $u$ is **path-constant** and identical to the instantaneous velocity $v_t(x)$. Mean flow matching achieves one-step sampling by exploiting this constancy, but the *functional form* of the vector field is identical to flow matching.

Thank you for the correction—this is an important nuance in the Gaussian case! Would you like me to derive the conditional expectation $\mathbb{E}[\mathbf{x}_1 - \mathbf{x}_0 \mid \mathbf{x}_t]$ step-by-step?








## Reference

MIT 6.S184: Flow Matching and Diffusion Models https://www.youtube.com/watch?v=GCoP2w-Cqtg&t=28s&ab_channel=PeterHolderrieth

Yaron Meta paper: [[2210.02747] Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

An Introduction to Flow Matching: https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html


## Appendix C

Compute Expectation** (Appendix C)

The conditional vector field should be: 
![[Pasted image 20250603200549.png]]

Since $u_t(x \mid \mathbf{x}_1) = \frac{\mathbf{x}_1 - x}{1-t}$ is linear in $\mathbf{x}_1$, and the posterior is Gaussian:
$$
u_t(x) = \mathbb{E}_{\mathbf{x}_1 \sim p_{1\mid t}(\mathbf{x}_1|x)}\left[\frac{\mathbf{x}_1 - x}{1-t} \right] = \frac{\overbrace{\mathbb{E}_{p_{1\mid t}}[\mathbf{x}_1 ]}^{\text{posterior mean}} - x}{1-t}
$$
$$
\boxed{\mathbf{u}_t(\mathbf{x}) = \frac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{2t^2-2t+1} = \frac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{\sigma^2_t}}
$$
### **Geometric Interpretation**
看下圖比較清楚：想像是一團 centered at (-10, 0) 的點，隨著時間往 (+10, 0) 移動的過程。綠色是 trace, 是每個點經過 vector field 被改變之後的 trace.  可以想像是微分方程的解。

$\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{u}_t(\boldsymbol{\mathbf{x}_t})$

- **For $t < 0.5$:**  
  The term $(2t-1)$ is negative, 所以是一個壓縮的流。
  
- **For $t > 0.5$:**  
  The term $(2t-1)$ is positive, 所以是一個膨脹的流。

![[Pasted image 20250604104124.png]]

The posterior mean is:
$$
\mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = x] = \frac{tx + (1-t)\boldsymbol{\mu}}{2t^2-2t+1}
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



One-sided flow matching?  Two sided flow matching?


## Reference

youtube video: next generation model
https://www.youtube.com/watch?v=swKdn-qT47Q&ab_channel=%E9%98%BF%E5%86%89

MIT Kaiming He
https://arxiv.org/abs/2505.13447


## Appendix A: Two Gaussian Case

Given the specific conditions $\sigma_0^2 = \sigma_1^2 = \sigma^2$, $\boldsymbol{\mu}_0 = -\boldsymbol{\mu}$, and $\boldsymbol{\mu}_1 = +\boldsymbol{\mu}$, the instantaneous vector field is:

$$
\mathbf{v}_t(\mathbf{x}) = \frac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{2t^2 - 2t + 1}
$$

The mean flow matching field $u(z_t, t_{\text{end}}, t)$ is defined as the time-average of $\mathbf{v}$ along the trajectory from $t$ to $t_{\text{end}}$:

$$
u(z_t, t_{\text{end}}, t) = \frac{1}{t_{\text{end}} - t} \int_t^{t_{\text{end}}} \mathbf{v}_s(\mathbf{x}_s)  ds
$$

where the trajectory $\mathbf{x}_s$ starting from $z_t$ at time $t$ is given by:

$$
\mathbf{x}_s = \boldsymbol{\mu}_s + \frac{\sigma_s}{\sigma_t} (z_t - \boldsymbol{\mu}_t)
$$

with:
- $\boldsymbol{\mu}_s = (2s - 1) \boldsymbol{\mu}$
- $\sigma_s = \sigma \sqrt{2s^2 - 2s + 1}$
- $\boldsymbol{\mu}_t = (2t - 1) \boldsymbol{\mu}$
- $\sigma_t = \sigma \sqrt{2t^2 - 2t + 1}$

Substitute $\mathbf{v}_s(\mathbf{x}_s)$ and simplify:

$$
\mathbf{v}_s(\mathbf{x}_s) = \frac{(2s-1)\mathbf{x}_s + \boldsymbol{\mu}}{2s^2 - 2s + 1}
$$

Using $\mathbf{x}_s = \boldsymbol{\mu}_s + \frac{\sigma_s}{\sigma_t} (z_t - \boldsymbol{\mu}_t)$ and $\boldsymbol{\mu}_s = (2s-1)\boldsymbol{\mu}$:

$$
\mathbf{v}_s(\mathbf{x}_s) = 2\boldsymbol{\mu} + \frac{(2s-1)}{\sqrt{2s^2 - 2s + 1} \sqrt{2t^2 - 2t + 1}} (z_t - \boldsymbol{\mu}_t)
$$

The integral becomes:

$$
\int_t^{t_{\text{end}}} \mathbf{v}_s(\mathbf{x}_s)  ds = 2\boldsymbol{\mu} (t_{\text{end}} - t) + \frac{1}{\sqrt{2t^2 - 2t + 1}} (z_t - \boldsymbol{\mu}_t) \left[ \sqrt{2s^2 - 2s + 1} \right]_t^{t_{\text{end}}}
$$

Evaluating the definite integral:

$$
\int_t^{t_{\text{end}}} \mathbf{v}_s(\mathbf{x}_s)  ds = 2\boldsymbol{\mu} (t_{\text{end}} - t) + \frac{\sqrt{2t_{\text{end}}^2 - 2t_{\text{end}} + 1} - \sqrt{2t^2 - 2t + 1}}{\sqrt{2t^2 - 2t + 1}} (z_t - \boldsymbol{\mu}_t)
$$

Divide by $t_{\text{end}} - t$:

$$
u(z_t, t_{\text{end}}, t) = 2\boldsymbol{\mu} + \frac{\sqrt{2t_{\text{end}}^2 - 2t_{\text{end}} + 1} - \sqrt{2t^2 - 2t + 1}}{\sqrt{2t^2 - 2t + 1} \cdot (t_{\text{end}} - t)} (z_t - \boldsymbol{\mu}_t)
$$

Substituting $\boldsymbol{\mu}_t = (2t - 1) \boldsymbol{\mu}$:

$$
\boxed{u(z_t, t_{\text{end}}, t) = 2\boldsymbol{\mu} + \dfrac{\sqrt{2t_{\text{end}}^2 - 2t_{\text{end}} + 1} - \sqrt{2t^2 - 2t + 1}}{\sqrt{2t^2 - 2t + 1} \cdot (t_{\text{end}} - t)} \left( z_t - (2t - 1) \boldsymbol{\mu} \right)}
$$

where:
- $z_t$ is the position at time $t$,
- $t_{\text{end}}$ is the end time,
- $t$ is the current time,
- $\boldsymbol{\mu}$ is a constant vector,
- $\sigma$ is a constant scalar (though it cancels out in the expression).

This expression simplifies to the general form derived earlier when $\sigma_0^2 = \sigma_1^2$, but is presented here explicitly for the given parameters.