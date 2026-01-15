---
title: Math AI - ODE Relationship to Thermodynamics
date: 2025-05-26 23:10:08
typora-root-url: ../../allenlu2009.github.io
categories:
  - Science
tags:
  - Fisher
  - Information
  - Math
  - Math_AI
  - Physics
  - Score
  - Thermodynamics
---





### 幾個常用公式：

**Conservation of probability:**

$$
\frac{d}{dt} \int p(x,t)\, dx = \int \frac{\partial p(x,t)}{\partial t}\, dx  = 0$$
$$
\mathbb{E}_{p} \left[ \frac{\partial}{\partial t} \log p(x,t) \right] = \int p(x,t) \cdot \frac{\partial}{\partial t} \log p(x,t)\, dx = \int \frac{\partial p(x,t)}{\partial t}\, dx = 0
$$
**Entropy 和時間變化：**
$$
H(t) = - \mathbb{E}_p\left[\log p(x,t)\right] = -\int p(x,t)\log p(x,t)dx > 0
$$
$$\begin{aligned}
\frac{d H(t)}{dt} &= -\int \frac{\partial p(x,t)}{\partial t}\log p(x,t)dx \\
&= \int \left[\nabla \cdot[\boldsymbol{u}(x,t)\, p(x,t)]-D(t) \Delta  p(x,t)\right] \log p(x,t)dx \\
&= \underbrace{\mathbb{E}[\nabla \cdot \boldsymbol{u}(x,t)]}_{\text{Drift contribution}} + \underbrace{D(t) I(p)}_{\text{Diffusion contribution}},
\end{aligned}$$
**Fisher information** $I(p) = \int p \|\nabla \log p\|^2 dx\, \ge 0$ .



### Fokker-Planck 偏微分方程

在 generative AI 的 diffusion process 或是 flow method:  守恆量是機率 (任意時間點的機率和為 1) $\Phi = p(x, t)$，沒有 source $S=0$.   不過一般物理的 diffusion 是從高濃度向低濃度，但是 probability 卻是從低機率 diffuse 到高機率。可以等價視爲負 diffusion constant:  $D = -\Gamma$ ，如果假設是 isotropic diffusion, $D(t)$ 和位置無關，可以和時間有關 (noise scheduling)。一般寫成：
$$
\frac{\partial p(x,t)}{\partial t}=-\nabla \cdot[\mathbf{u}(x,t)\, p(x,t)]+\nabla \cdot[D(t) \nabla p(x,t)] = -\nabla \cdot[\mathbf{u}(x,t)\, p(x,t)]+D(t) \Delta  p(x,t)  
$$
另一個表示

$$\begin{aligned}
\frac{\partial\log p(x,t)}{\partial t}&= \frac{1}{p(x,t)}\frac{\partial p(x,t)}{\partial t}\\
&=  -\frac{\nabla \cdot[\mathbf{u}(x,t)\, p(x,t)]}{p(x,t)}+D(t) \frac{\Delta  p(x,t)}{p(x,t)}\\  
&=  -\nabla \cdot \mathbf{u}(x,t)-\mathbf{u}(x,t)\cdot \nabla\log p(x,t) +D(t) \frac{\Delta  p(x,t)}{p(x,t)}\\  
&=  -\mathbf{u}(x,t)\cdot \nabla\log p(x,t) \underbrace{- \nabla \cdot \mathbf{u}(x,t) +D(t)[\Delta \log p(x,t)] +  \| \nabla \log p(x,t) \|^2]}_{=\frac{d\log p(x,t)}{d t}}\\ 
\end{aligned}$$

### Fokker-Planck 全微分方程 (不常用，容易錯 use with caution)

還有一個表示法是利用全微分 ODE，兩者等價。

- 偏微分：$p$ 是 $x,t$ 的函數。對應 (Euler) 靜止觀察者視角
- 全微分：$p$ 是 $x(t), t$ 的函數，最終都是 $t$ 的函數。對應 (Lagrange) 隨著粒子或流 $x_t$ 移動視角

**偏微分和全微分的關係**
$$\frac{d f(\mathbf{x},t)}{dt} = \frac{\partial f(\mathbf{x},t)}{\partial t} + \frac{d\mathbf{x}}{dt}\cdot \nabla f(\mathbf{x},t) = (\frac{\partial}{\partial t} + \frac{d\mathbf{x}}{dt}\cdot \nabla) f(\mathbf{x},t) =  (\frac{\partial}{\partial t} + \mathbf{u}\cdot \nabla) f(\mathbf{x},t)$$

全微分可以把 $p(x,t)$ 改成 $\log p(x,t)$ ，見  Appendix A.
$$\begin{aligned}
\frac{d \log p(x,t)}{d t}&= -\nabla \cdot \mathbf{u}(x,t) +  \nabla\cdot [D(t)\nabla \log p(x,t)] + D(t) \| \nabla \log p(x,t) \|^2\\  
&= -\nabla \cdot \mathbf{u}(x,t) +  D(t)[\Delta \log p(x,t)] +  \| \nabla \log p(x,t) \|^2]
\end{aligned}$$
偏微分和全微分的關係如下：
$$\begin{aligned}
\frac{d \log p(x,t)}{d t}&= \frac{\partial \log p(x,t)}{\partial t} + \mathbf{u}(x,t) \cdot \nabla\log p(x,t)
\end{aligned}$$
另一個不常用的表示
$$\begin{aligned}
\frac{d p(x,t)}{d t}&= \frac{\partial p(x,t)}{\partial t} + \mathbf{u}(x,t) \cdot \nabla p(x,t)\\
&=-\nabla \cdot[\mathbf{u}(x,t)\, p(x,t)]+\nabla \cdot[D(t) \nabla p(x,t)] + \mathbf{u}(x,t) \cdot \nabla p(x,t)\\
&= -\nabla \cdot[\mathbf{u}(x,t)\, p(x,t)]+D(t) \Delta  p(x,t)  + \mathbf{u}(x,t) \cdot \nabla p(x,t)\\
&=  -[\nabla \cdot \mathbf{u}(x,t)]\, p(x,t)+D(t) \Delta  p(x,t) 
\end{aligned}$$


> [!Special Case: No Diffusion, Flow Only]
> 
> #### **對於 flow only, no diffusion, $D = 0$, 可以簡化成:**
> $$
> \frac{\partial p(x,t)}{\partial t}=-\nabla \cdot[\mathbf{u}(x,t)\, p(x,t)] \quad \text{ or }\frac{d p(x,t)}{d t}=-[\nabla \cdot \mathbf{u}(x,t)]\, p(x,t)
> $$
> $$
> \frac{d \log p(x,t)}{d t}= -\nabla \cdot \mathbf{u}(x,t)  \quad \text{ or }\frac{\partial \log p(x,t)}{\partial t}=- \nabla \cdot \mathbf{u}(x,t)-\mathbf{u}(x,t)\cdot \nabla\log p(x,t) 
> $$
> 

#### 微觀 sample $x_t$ "全微分" SDE

以上是以 ODE 的形式，代表（巨觀）平均 flow $p(x,t)$ 的形式。另一個則是 SDE，代表（微觀）**個別樣本隨機運動 $x_t$ 的形式。個別 sample 一定是 $t$ 的函數，因此是全微分表示法。**

最 general 的形式如下。這裏的 $\sigma(x_t, t)$ 是 "**incremental**" additive noise，我們之後會改成 $g(t)$. 因爲很容易和後面的 "**total**" additive Gaussian noise $\boldsymbol{x}_t = \boldsymbol{x}_0 + \sigma(t) \boldsymbol{z}_t$ 混淆！ 
$$
d \boldsymbol{x}_t = \mathbf{u}(\boldsymbol{x}_t, t) d t+\sigma(\boldsymbol{x}_t, t) d \boldsymbol{w}_t
$$
對應的 Fokker-Planck
$$
\frac{\partial p(x,t)}{\partial t}= -\nabla \cdot[\mathbf{u}(x,t)\, p(x,t)]+ \Delta [D(x, t) p(x,t)]\quad \text{ where }D(x,t) = \frac{\sigma^2(x,t)}{2}  
$$
我們改成 isotropic 形式， 把 $\sigma(\boldsymbol{x}_t, t)$ 改成 $g(t)$，$\boldsymbol{u}(\boldsymbol{x}_t,t)$ 變成 $\boldsymbol{f}(\boldsymbol{x}_t,t)$. 
$$
d \boldsymbol{x}_t = \boldsymbol{f}(\boldsymbol{x}_t, t) d t+g(t) d \boldsymbol{w}_t\quad \text{ where }D(t) = \frac{g^2(t)}{2}
$$
對應的偏微分 PDE：
$$
\frac{\partial p(x,t)}{\partial t}=-\nabla \cdot[\boldsymbol{f}(x,t)\, p(x,t)]+\frac{g^2(t)}{2} \Delta  p(x,t)  
$$
全微分 ODE：
$$
\frac{d \log p(x,t)}{d t}=-\nabla \cdot\boldsymbol{f}(x,t)+\frac{g^2(t)}{2} [ \Delta  \log p(x,t) + \|\nabla\log p(x,t)\|^2]  
$$



## Conservation of Probability

Appendix P 證明 Fokker-Planck equation probability conservation (sum to 1).
注意都是偏微分 “靜止觀察者的 global view",  不是全微分 "粒子移動的 view"

$$
\frac{d}{dt} \int p(x,t)\, dx = \int \frac{\partial p(x,t)}{\partial t}\, dx = 0
$$
證明是 drift 和 diffusion 的時間微分，分別為 0。代表各自影響的 probability conservation？
物理意義？應該不是說 40% probability 是 drift, 60% probability 是 diffusion 各自 conservation.   比較好的類比是把墨水滴在水中，同時攪動水產生 flow field。總墨水量不變 (總 probability conservation)。而且 diffusion 散開的墨水量和攪動 drift 散開的墨水量不變？
還是其實沒有意義，因為 drift and diffusion 的 probability conservation 都是假設無窮大範圍的面積分為 0.

$$
\int \nabla \cdot (\boldsymbol{u} p)\, dx = 0 
$$

$$
\int \Delta p\, dx = \int \nabla \cdot (\nabla p)\, dx = \oint_{\partial \Omega} \nabla p \cdot \hat{n}\, dS = 0
$$

注意全微分 “粒子移動的 view".   如果我是粒子而且在一個壓縮的 flow field ($\nabla\cdot u<0$)，整體積分會得到變多的粒子？

$$\begin{aligned}
\int\frac{d p(x,t)}{d t} dx&= \underbrace{\int \frac{\partial p(x,t)}{\partial t} dx}_{=0}+ \int \mathbf{u}(x,t) \cdot \nabla p(x,t)dx\\
&=  -\int[\nabla \cdot \mathbf{u}(x,t)]\, p(x,t) dx \ne 0
\end{aligned}$$
也就是說
$$
\frac{d}{dt} \int p(x,t)\, dx = \int \frac{\partial p(x,t)}{\partial t}\, dx \ne \int \frac{d p(x,t)}{d t}\, dx
$$

**完全等價的 conservation of probability：（Appendix Q)**

$$
\mathbb{E}_{p} \left[ \frac{\partial}{\partial t} \log p(x,t) \right] = \int p(x,t) \cdot \frac{\partial}{\partial t} \log p(x,t)\, dx = \int \frac{\partial p(x,t)}{\partial t}\, dx = 0
$$
而不是
$$
\frac{d}{dt} \log p(x(t), t) = \frac{\partial}{\partial t} \log p(x,t) + \nabla \log p(x,t) \cdot \frac{dx}{dt}
$$

So the expectation becomes:

$$
\mathbb{E}_p \left[ \frac{d}{dt} \log p(x(t),t) \right]
= \underbrace{\mathbb{E} \left[ \frac{\partial}{\partial t} \log p(x,t) \right]}_{=0 \text{ prob. conservation} } + \mathbb{E} \left[ \nabla \log p(x,t) \cdot \frac{dx}{dt} \right]
$$

$$
\mathbb{E}_p \left[ \frac{d}{dt} \log p(x(t),t) \right] = \mathbb{E}_p \left[ \boldsymbol{u}(x,t) \cdot \nabla \log p(x,t)  \right] \ne 0
$$




### Fokker-Planck 和 Entropy 的連結 (微觀熱力學)

我們看一下 entropy 的公式：
$$
H(t) = - \mathbb{E}_p\left[\log p(x,t)\right] = -\int p(x,t)\log p(x,t)dx > 0
$$
重點是 entropy 隨時間變化。
注意 (ChatGPT, Appendix O)： $\frac{dH}{dt}$ 是 global (Euler) view, 所以在和積分交換，是採用偏微分 (固定觀察者觀點) $\frac{\partial p}{\partial t}$ 而不是全微分 (移動粒子觀點) $\frac{dp}{dt}$.  這點我還是有一些疑問。

$$\begin{aligned}
\frac{d H(t)}{dt} &= -\int \left[\frac{\partial p(x,t)}{\partial t}\log p(x,t) + p(x,t) \frac{1}{p(x,t)}\frac{\partial p(x,t)}{\partial t}\right]dx\\
&= -\int \frac{\partial p(x,t)}{\partial t}\log p(x,t)dx - \int \frac{\partial p(x,t)}{\partial t}dx\\
&= -\int \frac{\partial p(x,t)}{\partial t}\log p(x,t)dx -  \frac{d}{d t}\underbrace{\int p(x,t) dx}_{=1}\\
&= -\int \frac{\partial p(x,t)}{\partial t}\log p(x,t)dx \\
&= \int \left[\nabla \cdot[\boldsymbol{f}(x,t)\, p(x,t)]-\frac{g^2(t)}{2} \Delta  p(x,t)\right] \log p(x,t)dx \\
\end{aligned}$$

**The drift term** 可以化簡成平均 drift divergence (Appendix M) 

$$
\int \nabla \cdot(\boldsymbol{f}(x,t)\, p(x,t)) \log p(x,t) \, dx
= \int (\nabla \cdot \boldsymbol{f}(x,t))\, p(x,t) \, dx = \mathbb{E}_p [\nabla \cdot \boldsymbol{f}(x,t)]
$$

**The diffusion term** 可以化簡成

$$
-\frac{g^2(t)}{2} \int \Delta p(x,t) \log p(x,t)\, dx = \frac{g^2(t)}{2} \int p(x,t) \left\|\nabla \log p(x,t)\right\|^2 dx
$$

**Fisher information** $I(p) = \int p \|\nabla \log p\|^2 dx\, \ge 0$ .

Combining both terms:

$$
\frac{d H(t)}{dt} = \int \nabla \cdot \boldsymbol{f}(x,t)\, p(x,t)\, dx + \frac{g^2(t)}{2} \int p(x,t) \left\|\nabla \log p(x,t)\right\|^2 dx
$$

$$
\frac{dH(t)}{dt} = \underbrace{\mathbb{E}[\nabla \cdot \boldsymbol{f}(x,t)]}_{\text{Drift contribution}} + \underbrace{\frac{g^2(t)}{2} I(p)}_{\text{Diffusion contribution}},
$$

where:
- $\mathbb{E}[\nabla \cdot \boldsymbol{f}(x,t)]$ is the expected divergence of the drift field $\boldsymbol{f}(x,t)$.
- $\frac{g^2(t)}{2} I(p)$ is the diffusion term, with $I(p) = \mathbb{E}\left[\|\nabla \log p(x,t)\|^2\right] \geq 0$ (Fisher information, always non-negative).

### Entropy 隨時間變化

#### **1. Diffusion term: $\frac{g^2(t)}{2} I(p)$**

* Always **non-negative**, since $g^2(t) \ge 0$ and Fisher information $I(p) \ge 0$.
* Represents the **entropy-increasing** effect of diffusion (spreading out the distribution).
* Strictly positive unless $p$ is uniform or a Dirac delta (infinite entropy case).

#### **2. Drift term: $\mathbb{E}[\nabla \cdot \boldsymbol{f}(x,t)]$**

* Can be **positive, negative, or zero**, depending on the vector field $\boldsymbol{f}(x,t)$.
* If $\nabla \cdot \boldsymbol{f} < 0$ (e.g., a contracting flow), this term decreases entropy. 一般發生在 reverse diffusion (training and sampling).
* If $\nabla \cdot \boldsymbol{f} > 0$ (e.g., expanding flow), this term increases entropy.  一般發生在 forward diffusion (training).

---

### **Overall sign of $\frac{dH}{dt}$?**

We **cannot assert the sign of $\frac{dH}{dt}$ in general**, because it depends on the **balance between drift and diffusion**:

* If **diffusion dominates** (large $g(t)$, or small $\nabla \cdot \boldsymbol{f}$), entropy increases: $\frac{dH}{dt} > 0$
* If **drift dominates**, and especially if it's compressive: $\frac{dH}{dt} < 0$
* If they balance: $\frac{dH}{dt} = 0$, which can happen in stationary cases

---
### **Special case: Pure diffusion (no drift)**

If $\boldsymbol{f}(x,t) = 0$, then:

$$
\frac{dH(t)}{dt} = \frac{g^2(t)}{2} I(p) \ge 0
$$

So **entropy always increases** unless $I(p) = 0$, uniform distribution — this is **consistent with the heat equation**, where a peaked distribution spreads out over time.


### **Special case: Pure Gaussian diffusion (no drift)**

$$
p(x) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$
Then the squared norm:

$$
\left(\frac{d}{dx} \log p(x)\right)^2 = \frac{(x - \mu)^2}{\sigma^4}
$$

Take the expectation under $p(x)$:

$$
I(p) = \int p(x) \left( \frac{x - \mu}{\sigma^2} \right)^2 dx = \frac{1}{\sigma^4} \int p(x) (x - \mu)^2 dx = \frac{1}{\sigma^4} \cdot \sigma^2 = \boxed{\frac{1}{\sigma^2}}
$$
Gaussian distribution entropy
$$
H(p) = \frac{1}{2}\log(2\pi \sigma^2) + \frac{1}{2}
$$


> [!Multivariate version:]
> 
> If $p(x,t) = \mathcal{N}(\mu(t), \Sigma(t)) \in \mathbb{R}^d$, then:
> 
> $$
> I(p) = \int p(x)\, \|\nabla \log p(x)\|^2 dx = \operatorname{Tr}(\Sigma^{-1})
> $$
> because:
> 
> $$
> \nabla \log p(x) = -\Sigma^{-1}(x - \mu), \quad \text{so } \|\nabla \log p\|^2 = (x - \mu)^T \Sigma^{-2} (x - \mu)
> $$
> 
> Then take expectation:
> 
> $$
> I(p) = \mathbb{E}[(x - \mu)^T \Sigma^{-2} (x - \mu)] = \operatorname{Tr}(\Sigma^{-2} \mathbb{E}[(x - \mu)(x - \mu)^T]) = \operatorname{Tr}(\Sigma^{-2} \Sigma) = \operatorname{Tr}(\Sigma^{-1})
> $$
> for isotropic diffusion $\Sigma(t) = \sigma^2(t) I$, then  $I(p) = \frac{d}{\sigma^2(t)}$.   這裏的 $d$ 是 dimension, 不是微分符號！
> 

Entropy 變化從 pure diffusion
$$
\frac{dH(t)}{dt} = \frac{g^2(t)}{2} I(p) = \frac{g^2(t)}{2\sigma^2(t)} = \frac{d\sigma^2(t)}{2 \sigma^2(t) dt} = \frac{1}{2}\frac{d\log\sigma^2(t)}{dt}
$$
直接計算 entropy 變化
$$
\frac{dH(t)}{dt} = \frac{1}{2}\frac{d \log (2\pi \sigma^2(t))}{dt} = \frac{1}{2}\frac{d\log\sigma^2(t)}{dt}
$$
一致。




---
### **Special case:  Divergence-Free (不可壓縮) Flow**

If $\nabla\cdot \boldsymbol{f}(x,t) = 0$, then:

$$
\frac{dH(t)}{dt} = \frac{g^2(t)}{2} I(p) \ge 0
$$

So **entropy always increases** unless $I(p) = 0$, uniform distribution — this is **consistent with the heat equation**, where a peaked distribution spreads out over time.


---

### **Special case: Deterministic flow (no diffusion)**

If $g(t) = 0$, then:

$$
\frac{dH(t)}{dt} = \mathbb{E}[\nabla \cdot \boldsymbol{f}(x,t)]
$$

This can be positive, negative, or 0 depending on whether the deterministic flow expands or contracts space.

### Special Case: Gradient Flow
If the drift is a gradient of a potential $\boldsymbol{f}(x,t) = -\nabla V(x)$, then:
$$
\nabla \cdot \boldsymbol{f}(x,t) = -\Delta V(x),
$$
and the sign depends on the Laplacian of $V$:
- If $\Delta V(x) > 0$ (subharmonic potential), $\mathbb{E}[\nabla \cdot \boldsymbol{f}] < 0$, which can lead to entropy decrease.
- If $\Delta V(x) < 0$ (superharmonic potential), $\mathbb{E}[\nabla \cdot \boldsymbol{f}] > 0$, increasing entropy.

### **Conclusion**

We **cannot generally say** whether $\frac{dH}{dt}$ is positive or negative **without knowing more about the drift and diffusion**. However:

* **Diffusion always increases entropy**
* **Drift can increase or decrease entropy**, depending on whether it compresses or expands probability mass.

This interplay is fundamental in stochastic processes and nonequilibrium thermodynamics.




## Appendix M

We will continue the derivation from:

$$
\frac{d H(t)}{dt} = \int \left[\nabla \cdot(\boldsymbol{f}(x,t)\, p(x,t)) - \frac{g^2(t)}{2} \Delta p(x,t)\right] \log p(x,t)\, dx
$$

We now handle the two terms in the integrand separately.

---

### **1. The drift term:**

$$
\int \nabla \cdot(\boldsymbol{f}(x,t)\, p(x,t)) \log p(x,t) \, dx
$$

Use **integration by parts** (divergence theorem) in reverse, assuming boundary terms vanish (e.g., decay at infinity):

$$
\int \nabla \cdot(\boldsymbol{f}\, p) \log p \, dx = -\int \boldsymbol{f}(x,t)\, p(x,t) \cdot \nabla \log p(x,t) \, dx
$$

Using the identity $\nabla \log p = \frac{\nabla p}{p}$, we simplify:

$$
= -\int \boldsymbol{f}(x,t) \cdot \nabla p(x,t) \, dx
$$

Now integrate by parts again, assuming boundary terms vanish:

$$
= \int \nabla \cdot \boldsymbol{f}(x,t)\, p(x,t) \, dx
$$

---

### **2. The diffusion term:**

$$
-\frac{g^2(t)}{2} \int \Delta p(x,t) \log p(x,t)\, dx
$$

We use integration by parts, with the identity:

$$
\int \Delta p \log p \, dx = -\int \frac{\|\nabla p\|^2}{p} \, dx = -\int p(x,t) \left\|\nabla \log p(x,t)\right\|^2 dx
$$

Hence, the diffusion contribution becomes:

$$
\frac{g^2(t)}{2} \int p(x,t) \left\|\nabla \log p(x,t)\right\|^2 dx
$$

This is the **Fisher information** $I(p) = \int p \|\nabla \log p\|^2 dx$.

---

### **Final expression:**

Combining both terms:

$$
\frac{d H(t)}{dt} = \int \nabla \cdot \boldsymbol{f}(x,t)\, p(x,t)\, dx + \frac{g^2(t)}{2} \int p(x,t) \left\|\nabla \log p(x,t)\right\|^2 dx
$$

Or more compactly:

$$
\boxed{
\frac{d H(t)}{dt} = \mathbb{E}[\nabla \cdot \boldsymbol{f}(x,t)] + \frac{g^2(t)}{2} I(p)
}
$$

where:

* $\mathbb{E}[\nabla \cdot \boldsymbol{f}]$ is the expected divergence of the drift,
* $I(p)$ is the Fisher information of the distribution $p(x,t)$.


## Appendix N

Great question. Let's analyze the sign of:

$$
\frac{dH(t)}{dt} = \mathbb{E}[\nabla \cdot \boldsymbol{f}(x,t)] + \frac{g^2(t)}{2} I(p)
$$

This tells us how the **differential entropy** $H(t) = -\int p(x,t) \log p(x,t) dx$ evolves over time, for a probability distribution $p(x,t)$ governed by a Fokker–Planck equation with drift $\boldsymbol{f}(x,t)$ and scalar diffusion coefficient $g(t)$.

---

### **Term-by-term analysis**

#### **1. Diffusion term: $\frac{g^2(t)}{2} I(p)$**

* Always **non-negative**, since $g^2(t) \ge 0$ and Fisher information $I(p) \ge 0$.
* Represents the **entropy-increasing** effect of diffusion (spreading out the distribution).
* Strictly positive unless $p$ is uniform or a Dirac delta (infinite entropy case).

#### **2. Drift term: $\mathbb{E}[\nabla \cdot \boldsymbol{f}(x,t)]$**

* Can be **positive, negative, or zero**, depending on the vector field $\boldsymbol{f}(x,t)$.
* If $\nabla \cdot \boldsymbol{f} < 0$ (e.g., a contracting flow), this term decreases entropy.
* If $\nabla \cdot \boldsymbol{f} > 0$ (e.g., expanding flow), this term increases entropy.

---

### **Overall sign of $\frac{dH}{dt}$?**

We **cannot assert the sign of $\frac{dH}{dt}$ in general**, because it depends on the **balance between drift and diffusion**:

* If **diffusion dominates** (large $g(t)$, or small $\nabla \cdot \boldsymbol{f}$), entropy increases: $\frac{dH}{dt} > 0$
* If **drift dominates**, and especially if it's compressive: $\frac{dH}{dt} < 0$
* If they balance: $\frac{dH}{dt} = 0$, which can happen in stationary cases

---

### **Special case: Pure diffusion (no drift)**

If $\boldsymbol{f}(x,t) = 0$, then:

$$
\frac{dH(t)}{dt} = \frac{g^2(t)}{2} I(p) \ge 0
$$

So **entropy always increases** — this is **consistent with the heat equation**, where a peaked distribution spreads out over time.

---

### **Special case: Deterministic flow (no diffusion)**

If $g(t) = 0$, then:

$$
\frac{dH(t)}{dt} = \mathbb{E}[\nabla \cdot \boldsymbol{f}(x,t)]
$$

This can be positive or negative depending on whether the deterministic flow expands or contracts space.

---

### **Conclusion**

We **cannot generally say** whether $\frac{dH}{dt}$ is positive or negative **without knowing more about the drift and diffusion**. However:

* **Diffusion always increases entropy**
* **Drift can increase or decrease entropy**, depending on whether it compresses or expands probability mass.

This interplay is fundamental in stochastic processes and nonequilibrium thermodynamics.


## Appendix O

Great question! The **correct expression** for the time derivative of **Shannon entropy** is:

$$
\boxed{
\frac{dH(t)}{dt} = -\int \frac{\partial p(x,t)}{\partial t} \log p(x,t)\, dx
}
$$

### Why not the material derivative $\frac{d p}{dt}$?

The **material derivative** $\frac{d p}{dt} = \frac{\partial p}{\partial t} + \mathbf{u} \cdot \nabla p$ describes how $p$ changes **along a moving particle's path** in a flow field $\mathbf{u}(x,t)$. It's used in **fluid mechanics** or **stochastic processes** when tracking individual trajectories.

However, **entropy** is a **global functional** of the probability density:

$$
H(t) = -\int p(x,t) \log p(x,t) \, dx
$$

So when you take its time derivative, you apply the **chain rule** to the integrand:

$$
\frac{dH}{dt} = -\int \left( \frac{\partial p}{\partial t} \log p + \frac{\partial p}{\partial t} \right) dx
$$

But since $\int \frac{\partial p}{\partial t} dx = \frac{d}{dt} \int p(x,t) dx = 0$, that second term vanishes.

So you're left with:

$$
\frac{dH}{dt} = -\int \frac{\partial p(x,t)}{\partial t} \log p(x,t) \, dx
$$

### ✅ Therefore:

Use the **partial derivative $\partial p / \partial t$**, **not** the material derivative $d p / dt$.

---

### 檢查如果用全微分錯誤的答案

$$\begin{aligned}
\frac{d H(t)}{dt} &= -\int \left[\frac{d p(x,t)}{d t}\log p(x,t) + p(x,t) \frac{1}{p(x,t)}\frac{d p(x,t)}{d t}\right]dx\\
&= -\int \frac{d p(x,t)}{d t}\log p(x,t)dx - \int \frac{d p(x,t)}{d t}dx\\
&= -\int \frac{d p(x,t)}{d t}\log p(x,t)dx -  \frac{d}{d t}\underbrace{\int p(x,t) dx}_{=1}\\
&= -\int \frac{d p(x,t)}{d t}\log p(x,t)dx \\
&= \int \left[\nabla \cdot[\boldsymbol{f}(x,t)]\, p(x,t)-\frac{g^2(t)}{2} \Delta  p(x,t)\right] \log p(x,t)dx \\
&= \underbrace{\int\nabla \cdot[\boldsymbol{f}(x,t)]\, p(x,t)\log p(x,t)}_{\text{no simplification}} + \underbrace{\frac{g^2(t)}{2} I(p)}_{\text{Diffusion contribution}}\\
&\ne \frac{\partial H}{\partial t}
\end{aligned}$$
## Appendix P

To **prove conservation of probability**, you want to show that the **total probability mass stays constant over time** — i.e.,

$$
\boxed{
\frac{d}{dt} \int p(x,t)\, dx = 0
}
$$

You are given the **Fokker–Planck equation** (also called the forward Kolmogorov equation):

$$
\frac{\partial p(x,t)}{\partial t} = -\nabla \cdot \left[ \boldsymbol{f}(x,t)\, p(x,t) \right] + \frac{g^2(t)}{2} \Delta p(x,t)
$$

---

### ✅ Step-by-step Proof

We start by computing the time derivative of the total probability mass:

$$
\frac{d}{dt} \int p(x,t)\, dx = \int \frac{\partial p(x,t)}{\partial t}\, dx
$$

Now substitute the right-hand side of the Fokker–Planck equation:

$$
\int \left[ -\nabla \cdot (\boldsymbol{f} p) + \frac{g^2(t)}{2} \Delta p \right] dx
$$

Break this into two terms:

$$
- \int \nabla \cdot (\boldsymbol{f} p)\, dx + \frac{g^2(t)}{2} \int \Delta p\, dx
$$

We now evaluate each term using the **divergence theorem** (or integration by parts):

---

### 🧮 Term 1: $\int \nabla \cdot (\boldsymbol{f} p)\, dx$

By the divergence theorem:

$$
\int \nabla \cdot (\boldsymbol{f} p)\, dx = \oint_{\partial \Omega} (\boldsymbol{f} p) \cdot \hat{n}\, dS
$$

If we assume that $p(x,t) \to 0$ fast enough as $\|x\| \to \infty$ (e.g., for Gaussian-like distributions), then the surface integral at infinity vanishes:

$$
\boxed{ \int \nabla \cdot (\boldsymbol{f} p)\, dx = 0 }
$$

---

### 🧮 Term 2: $\int \Delta p\, dx$

Again using integration by parts or the divergence theorem:

$$
\int \Delta p\, dx = \int \nabla \cdot (\nabla p)\, dx = \oint_{\partial \Omega} \nabla p \cdot \hat{n}\, dS
$$

This also vanishes under the assumption that $\nabla p \to 0$ at the boundary (i.e., $p$ is smooth and decays rapidly):

$$
\boxed{ \int \Delta p\, dx = 0 }
$$

---

### ✅ Final Result:

Putting both terms together:

$$
\frac{d}{dt} \int p(x,t)\, dx = 0
$$

This proves **conservation of probability mass**.

---

### 📌 Assumptions:

This holds **under mild regularity assumptions**, such as:

* $p(x,t)$ and $\boldsymbol{f}(x,t)$ are smooth,
* $p(x,t) \to 0$ and $\nabla p(x,t) \to 0$ fast enough as $\|x\| \to \infty$ (e.g., exponential or Gaussian decay),
* No sources or sinks (mass creation or destruction).

These are standard in Fokker–Planck settings.

Let me know if you want this done for bounded domains with boundary conditions too.

